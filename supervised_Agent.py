import tensorflow as tf
from encoder import encode_seq
from decoder import pointer_for_sl
from tensorflow.contrib import distributions
from layers import embed_seq
from utils import vectorize_indices, permute


class Actor(object):

    def __init__(self, config):
        # Configuration
        self.version = config.version
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.graph_dimension  # input sequence length (number of cities)
        self.dimension = config.dimension  # dimension of a city (coordinates)

        # Network config
        self.input_embed = config.input_embed  # dimension of embedding space
        self.num_neurons = config.num_neurons  # dimension of hidden states (encoder)
        self.num_stacks = config.num_stacks  # encoder num stacks
        self.num_heads = config.num_heads  # encoder num heads
        self.query_dim = config.query_dim  # decoder query space dimension
        self.num_units = config.num_units  # dimension of attention product space (decoder and critic)
        self.num_neurons_critic = config.num_neurons_critic  # critic n-1 layer num neurons
        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer

        # Training config (actor and critic)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # actor global step
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # critic global step
        self.init_B = config.init_B  # critic initial baseline
        self.lr_start = config.lr_start  # initial learning rate
        self.lr_decay_step = config.lr_decay_step  # learning rate decay step
        self.lr_decay_rate = config.lr_decay_rate  # learning rate decay rate
        self.is_training = config.is_training  # switch to False if test mode
        self.C = config.C
        self.temperature = tf.placeholder("float", 1)  # config.temperature

        # instance variables to save
        self.logits1, self.logits2, self.idx, self.encoded_ref, \
        self.inter_city_distances, self.permutations, self.log_prob_mean, self.entropies_mean, \
        self.reward, self.predictions, self.loss, self.loss_2, \
        self.trn_op1, self.trn_op2, self.trn_op3, self.v_g, self.v = (None for _ in range(17))

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [None, self.max_length, self.dimension], name="input_coordinates")

        if 'graph' in self.version:
            self.graph_structure = tf.placeholder(tf.float32,
                                                  [None, self.max_length, self.max_length], name='adj_matrix')
        else:
            self.graph_structure = None

        if self.is_training:
            self.optimal_tour = tf.placeholder(tf.int32, [None, self.max_length], name="optimal_tour")
            self.next = permute(self.optimal_tour, 1)
            self.next_next = permute(self.optimal_tour, 2)
        else:
            self.current = tf.placeholder(tf.int32, [None], name="current_position")
            self.previous = tf.placeholder(tf.int32, [None], name="previous_position")
            self.previous2 = tf.placeholder(tf.int32, [None], name="previous2_position")

        with tf.variable_scope("actor"): self.encode_decode()
        if self.is_training:
            with tf.variable_scope("environment"): self.build_reward()
            with tf.variable_scope("optimizer"): self.build_optim()
        self.merged = tf.summary.merge_all()

    def encode_decode(self):
        embedding = embed_seq(input_seq=self.input_, from_=self.dimension, to_=self.input_embed,
                              is_training=self.is_training, BN=True,
                              initializer=tf.contrib.layers.xavier_initializer())

        encoding = encode_seq(input_seq=embedding, graph_tensor=self.graph_structure, input_dim=self.input_embed,
                              num_stacks=self.num_stacks, num_heads=self.num_heads,
                              num_neurons=self.num_neurons, is_training=self.is_training, version=self.version)
        n_hidden = encoding.get_shape().as_list()[2]  # input_embed

        if not self.is_training:
            encoding = tf.tile(encoding, [self.batch_size, 1, 1])

        with tf.variable_scope("Decoder"):
            W_ref = tf.get_variable("W_ref", [1, n_hidden, self.num_units], initializer=self.initializer)
            encoded_ref = tf.nn.conv1d(encoding, W_ref, 1, "VALID")  # [Batch size, seq_length, n_hidden]
            self.encoded_ref = tf.expand_dims(encoded_ref, 1)

            # initialize weights for pointer
            W_q = tf.get_variable("W_q", [self.query_dim, self.num_units], initializer=self.initializer)
            self.v = tf.get_variable("v", [self.num_units], initializer=self.initializer)
            # self.v_g = tf.get_variable("v_g", [1], initializer=self.initializer)

            # initialize weights for query
            W_1 = tf.get_variable("W_1", [n_hidden, self.query_dim], initializer=tf.initializers.he_normal())
            W_2 = tf.get_variable("W_2", [n_hidden, self.query_dim], initializer=tf.initializers.he_normal())
            W_3 = tf.get_variable("W_3", [n_hidden, self.query_dim], initializer=tf.initializers.he_normal())

            if self.is_training:
                # initialize query vectors
                indices_tours = vectorize_indices(self.optimal_tour, self.batch_size, for_sl=True)
                tour_embeddings = tf.gather_nd(encoding, indices_tours)
                tour_embeddings = tf.reshape(tour_embeddings, [-1, self.max_length, n_hidden])

                present_city = tf.tensordot(tour_embeddings, W_1, axes=1)
                previous_city = tf.tensordot(tour_embeddings, W_2, axes=1)
                previous_city = permute(previous_city, -1)
                previous2_city = tf.tensordot(tour_embeddings, W_3, axes=1)
                previous2_city = permute(previous2_city, -2)

                # query = tf.nn.relu(present_city + previous_city + previous2_city)
                query = tf.nn.relu(present_city + previous_city)

                self.logits1 = pointer_for_sl(encoded_ref=self.encoded_ref, query=query,
                                              W_q=W_q, v=self.v, C=self.C,
                                              temperature=self.temperature)

                prob = distributions.Categorical(self.logits1)  # logits = masked_scores
                self.idx = prob.sample()
                idx_ = vectorize_indices(self.idx, self.batch_size, for_sl=True)
                next_city = tf.gather_nd(encoding, idx_)  # update trajectory (state)
                next_city = tf.reshape(next_city, [-1, self.max_length, n_hidden])

                present_city = tf.tensordot(next_city, W_1, axes=1)
                previous_city = tf.tensordot(tour_embeddings, W_2, axes=1)
                previous2_city = tf.tensordot(tour_embeddings, W_3, axes=1)
                previous2_city = permute(previous2_city, -1)
                # query2 = tf.nn.relu(present_city + previous_city + previous2_city)
                query2 = tf.nn.relu(present_city + previous_city)

                self.logits2 = pointer_for_sl(encoded_ref=self.encoded_ref, query=query2,
                                              W_q=W_q, v=self.v, C=self.C,
                                              temperature=self.temperature)
                self.log_prob_mean = tf.reduce_mean(prob.log_prob(self.idx))
                self.entropies_mean = tf.reduce_mean(prob.entropy())
                tf.summary.scalar('log_prob_mean', self.log_prob_mean)
                tf.summary.scalar('entropies_mean', self.entropies_mean)

            else:
                indices_current = vectorize_indices(self.current, self.batch_size)
                encoding_current = tf.gather_nd(encoding, indices_current)
                present_city = tf.tensordot(encoding_current, W_1, axes=1)

                indices_previous = vectorize_indices(self.previous, self.batch_size)
                encoding_previous = tf.gather_nd(encoding, indices_previous)
                previous_city = tf.tensordot(encoding_previous, W_2, axes=1)

                indices_previous2 = vectorize_indices(self.previous2, self.batch_size)
                encoding_previous2 = tf.gather_nd(encoding, indices_previous2)
                previous2_city = tf.tensordot(encoding_previous2, W_3, axes=1)

                # query = tf.nn.relu(present_city + previous_city + previous2_city)
                query = tf.nn.relu(present_city + previous_city)

                self.logits1 = pointer_for_sl(encoded_ref=self.encoded_ref, query=query,
                                              W_q=W_q, v=self.v, C=self.C,
                                              temperature=self.temperature, training=False)




    def build_optim(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):  # Update moving_mean and moving_variance for BN
            lr1 = tf.train.natural_exp_decay(learning_rate=self.lr_start, global_step=self.global_step,
                                             decay_steps=self.lr_decay_step,
                                             decay_rate=self.lr_decay_rate, staircase=False,
                                             name="learning_rate1")  # learning rate actor
            tf.summary.scalar('lr', lr1)
            opt1 = tf.train.AdamOptimizer(learning_rate=lr1)  # Optimizer
            with tf.name_scope('first_step'):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.next,
                                                                           logits=self.logits1)
                gvs1 = opt1.compute_gradients(self.loss)  # gradients
                capped_gvs1 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs1 if
                               grad is not None]  # L2 clip
                self.trn_op1 = opt1.apply_gradients(grads_and_vars=capped_gvs1,
                                                    global_step=self.global_step)  # minimize op actor

            with tf.name_scope('second_step'):
                self.loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.next_next,
                                                                             logits=self.logits2)
                gvs2 = opt1.compute_gradients(self.loss_2)  # gradients
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if
                               grad is not None]  # L2 clip
                self.trn_op2 = opt1.apply_gradients(grads_and_vars=capped_gvs2,
                                                    global_step=self.global_step)  # minimize op actor

    def build_reward(self):  # reorder input % tour and return tour length (euclidean distance)
        self.permutations = tf.stack(
            [tf.tile(tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), 1), [1, self.max_length]),
             self.idx], 2)
        if self.is_training == True:
            self.ordered_input_ = tf.gather_nd(self.input_, self.permutations)
        else:
            self.ordered_input_ = tf.gather_nd(tf.tile(self.input_, [self.batch_size, 1, 1]), self.permutations)
        self.ordered_input_ = tf.transpose(self.ordered_input_, [2, 1,
                                                                 0])  # [features, seq length +1, batch_size]   Rq: +1 because end = start

        ordered_x_ = self.ordered_input_[0]  # ordered x, y coordinates [seq length +1, batch_size]
        ordered_y_ = self.ordered_input_[1]  # ordered y coordinates [seq length +1, batch_size]
        delta_x2 = tf.transpose(tf.square(ordered_x_[1:] - ordered_x_[:-1]),
                                [1, 0])  # [batch_size, seq length]        delta_x**2
        delta_y2 = tf.transpose(tf.square(ordered_y_[1:] - ordered_y_[:-1]),
                                [1, 0])  # [batch_size, seq length]        delta_y**2

        inter_city_distances = tf.sqrt(
            delta_x2 + delta_y2)  # sqrt(delta_x**2 + delta_y**2) this is the euclidean distance between each city: depot --> ... ---> depot      [batch_size, seq length]
        self.distances = tf.reduce_sum(inter_city_distances, axis=1)  # [batch_size]
        self.reward = tf.cast(self.distances, tf.float32)  # define reward from tour length
        tf.summary.scalar('reward_mean', tf.reduce_mean(self.reward))

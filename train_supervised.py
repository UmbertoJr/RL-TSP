import tensorflow as tf
from tqdm import tqdm
import numpy as np
from supervised_Agent import Actor
from configuration import get_config, name_model
from dataGenerator import DataGenerator
import os
from utils import count_no_weights, create_MST_graph, create_graph
import pickle

config = get_config()
config.graph_dimension = 50

config_str = 'TSP_' + str(config.dimension) + 'D_' + 'TSP' + str(config.graph_dimension) + '_b' + str(
    config.batch_size) + '_e' + str(config.input_embed) + '_n' + str(config.num_neurons) + '_s' + str(
    config.num_stacks) + '_h' + str(config.num_heads) + '_q' + str(config.query_dim) + '_u' + str(
    config.num_units) + '_c' + str(config.num_neurons_critic) + '_lr' + str(config.lr_start) + '_d' + str(
    config.lr_decay_step) + '_' + str(config.lr_decay_rate) + '_T' + str(config.temperature) + '_steps' + str(
    config.nb_steps) + '_i' + str(config.init_B)

nome = name_model().two_sides
dir_actor: str = 'SL' + nome + config_str
print(dir_actor)
print(config.version)

tf.reset_default_graph()
actor = Actor(config)  # Build graph

# Save & restore all the variables.
variables_to_save_all = [v for v in tf.global_variables() if 'Adam' not in v.name]
saver_all = tf.train.Saver(var_list=variables_to_save_all, keep_checkpoint_every_n_hours=1.0)


########################################## TRAIN #################################

dataset = DataGenerator()

tf.set_random_seed(123)
tf.random.set_random_seed(123)
temperature = 1.0
last = 10000


save_path = "saveSL/" + dir_actor
if not os.path.exists(save_path):
    os.makedirs(save_path)


restore_path = "saveSL/" + dir_actor + "/actor.ckpt"
with tf.Session() as sess:  # start session+ "next"
    count_no_weights()
    sess.run(tf.global_variables_initializer())  # run initialize op

    # saver_all.restore(sess, restore_path)
    writer = tf.summary.FileWriter('summarySL/' + dir_actor, sess.graph)  # summary writer

    for i in tqdm(range(10000, 10000 + config.nb_steps)):  # Forward pass & train step
        seed_ = i + 1
        input_batch, dist_batch = dataset.create_batch(actor.batch_size, actor.max_length,
                                                       actor.dimension, seed=seed_)
        if i % 100 == 0:
            with open("supervised_data/batch_seed_" + str(i) + "_" + str(i + 100) + ".pkl", 'rb') as handle:
                optimal_tour_dic = pickle.load(handle)

        optimal_tours = np.array(optimal_tour_dic[seed_])
        feed = {actor.input_: input_batch, actor.optimal_tour: optimal_tours,
                actor.temperature: np.array([temperature])}  # get feed dict   actor.predictions: real_lengths

        if actor.version == 'graph':
            graph_struct = create_graph(dist_batch)
            # graph_struct = create_MST_graph(dist_batch)
            feed[actor.graph_structure] = graph_struct

        _,  loss_first, loss2_first, reward_first = sess.run([actor.trn_op1, actor.loss, actor.loss_2, actor.reward],
                                                             feed_dict=feed)

        feed[actor.optimal_tour] = optimal_tours[:, ::-1]

        _, summary, \
        v, loss, loss2, reward, \
        logits1, next_sampled, indices, \
        entropy, log_probs, = sess.run([actor.trn_op1, actor.merged,
                                        actor.v, actor.loss, actor.loss_2, actor.reward,
                                        actor.logits1, actor.idx, actor.encoded_ref,
                                        actor.entropies_mean, actor.log_prob_mean],
                                       feed_dict=feed)

        if i % 100 == 0:
            print('\nv_mean', np.mean(v))
            print('loss first\t', np.mean(loss_first))
            print('loss\t', np.mean(loss))
            print('loss2 first\t', np.mean(loss2))
            print('loss2\t', np.mean(loss2))
            print('entropy mean', entropy)
            print('log_probs', log_probs)
            print('reward mean', np.mean(reward))
            writer.add_summary(summary, i)
            saver_all.save(sess, save_path + "/actor.ckpt")  # save the variables to disk

    saver_all.save(sess, save_path + "/actor.ckpt")  # save the variables to disk
    print("Training COMPLETED! Model saved in file: %s" % save_path)

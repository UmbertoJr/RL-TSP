import tensorflow as tf


def pointer(encoded_ref, query, mask, W_q, v, C=10., temperature=1.0):
    """
    From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    predict a distribution over next decoder input
    @param encoded_ref:
    @param query:
    @param mask:
    @param W_q:
    @param v:
    @param C:
    @param temperature:
    @return:
    """
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1)  # [Batch size, 1, n_hidden]

    scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])  # [Batch size, seq_length]
    scores = C * tf.tanh(scores / temperature)  # control entropy
    masked_scores = tf.clip_by_value(scores - 100000000. * mask, -100000000., 100000000.)  # [Batch size, seq_length]
    return masked_scores


def pointer_with_dist(encoded_ref, query, W_q, v, graph_euc_dist_query, v_g,
                      C=10., temperature=1.0, method='no', mask=None):
    """

    @param encoded_ref:
    @param query:
    @param mask:
    @param W_q:
    @param v:
    @param graph_euc_dist_query:
    @param v_g:
    @param C:
    @param temperature:
    @param method:
    @return:
    """
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1)  # [Batch size, 1, n_hidden]

    # scores = tf.reduce_sum(v * tf.square(encoded_ref - encoded_query), [-1])  # [Batch size, seq_length]
    scores = encoded_ref + encoded_query  # [Batch size, seq_length]

    if method == 'linear':
        graph_euc_dist_query = tf.expand_dims(graph_euc_dist_query, [-1])
        scores += v_g * graph_euc_dist_query
    elif method == 'non_linear':
        scores += mlp_ff(graph_euc_dist_query, encoded_query, encoded_ref, v_g)

    scores = v * tf.tanh(scores)
    scores = tf.reduce_sum(scores, [-1])
    scores = C * tf.tanh(scores / temperature)  # control entropy
    masked_scores = tf.clip_by_value(scores - 100000000. * mask, -100000000., 100000000.)  # [Batch size, seq_length]
    return masked_scores


def mlp_ff(distances,  query, ref, v_g):
    lo = tf.nn.relu(tf.matmul(distances, v_g[1]))
    return


def full_glimpse(ref, from_, to_, initializer=tf.contrib.layers.xavier_initializer()):
    """
    From a query [Batch size, n_hidden], glimpse at a set of reference vectors (ref) [Batch size, seq_length, n_hidden]
    @param ref:
    @param from_:
    @param to_:
    @param initializer:
    @return:
    """
    with tf.variable_scope("glimpse"):
        W_ref_g = tf.get_variable("W_ref_g", [1, from_, to_], initializer=initializer)
        v_g = tf.get_variable("v_g", [to_], initializer=initializer)
        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, W_ref_g, 1, "VALID",
                                     name="encoded_ref_g")  # [Batch size, seq_length, n_hidden]
        scores_g = tf.reduce_sum(v_g * tf.tanh(encoded_ref_g), [-1], name="scores_g")  # [Batch size, seq_length]
        attention_g = tf.nn.softmax(scores_g, name="attention_g")
        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g, 2))
        glimpse = tf.reduce_sum(glimpse, 1)
        return glimpse


def pointer_for_sl(encoded_ref, query, W_q, v, C=10., temperature=1.0, training=True):
    """
    From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    predict a distribution over next decoder input
    @param encoded_ref:
    @param query:
    @param W_q:
    @param v:
    @param C:
    @param temperature:
    @return:
    """
    if training:
        encoded_query = tf.expand_dims(tf.tensordot(query, W_q, axes=1), 2)  # [Batch size, max_len, 1, n_hidden]
    else:
        encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1)  # [Batch size, 1, n_hidden]

    scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query), [-1])  # [Batch size, seq_length]
    scores = C * tf.tanh(scores / temperature)  # control entropy
    return scores


def pointer_with_mst(encoded_ref, query, W_q, v, adj_mst, v_g,
                     C=10., temperature=1.0, method='no', mask=None):
    """

    @param encoded_ref:
    @param query:
    @param mask:
    @param W_q:
    @param v:
    @param graph_euc_dist_query:
    @param v_g:
    @param C:
    @param temperature:
    @param method:
    @return:
    """
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1)  # [Batch size, 1, n_hidden]

    # scores = tf.reduce_sum(v * tf.square(encoded_ref - encoded_query), [-1])  # [Batch size, seq_length]
    scores = v * tf.tanh(encoded_ref + encoded_query)  # [Batch size, seq_length]

    if method == 'linear':
        adj_mst = tf.expand_dims(adj_mst, [-1])
        scores += v_g * adj_mst

    scores = tf.reduce_sum(scores, [-1])
    scores = C * tf.tanh(scores / temperature)  # control entropy

    masked_scores = tf.clip_by_value(scores - 100000000. * mask, -100000000., 100000000.)  # [Batch size, seq_length]
    return masked_scores


def pointer_with_mst_and_dist(encoded_ref, query, W_q, v, adj_mst, graph_euc_dist_query, v_g, v_g2,
                     C=10., temperature=1.0, method='no', mask=None):
    """

    @param encoded_ref:
    @param query:
    @param mask:
    @param W_q:
    @param v:
    @param graph_euc_dist_query:
    @param v_g:
    @param C:
    @param temperature:
    @param method:
    @return:
    """
    encoded_query = tf.expand_dims(tf.matmul(query, W_q), 1)  # [Batch size, 1, n_hidden]

    # scores = tf.reduce_sum(v * tf.square(encoded_ref - encoded_query), [-1])  # [Batch size, seq_length]
    scores = v * tf.tanh(encoded_ref + encoded_query)  # [Batch size, seq_length]

    if method == 'linear':
        adj_mst = tf.expand_dims(adj_mst, [-1])
        scores += v_g * adj_mst
        graph_euc_dist_query = tf.expand_dims(graph_euc_dist_query, [-1])
        scores += v_g2 * graph_euc_dist_query

    scores = tf.reduce_sum(scores, [-1])
    scores = C * tf.tanh(scores / temperature)  # control entropy

    masked_scores = tf.clip_by_value(scores - 100000000. * mask, -100000000., 100000000.)  # [Batch size, seq_length]
    return masked_scores

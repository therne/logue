
import tensorflow as tf
from utils.nn import linear
from .tdnn import TDNN

def embed_characters(input, vocab_size, embed_dim=40, scope=None, reuse=None,
             use_batch_norm=True, use_highway=True, highway_layers=2):
    """ Character-level embedding """

    with tf.variable_scope(scope or 'Embedder') as scope:
        if reuse: scope.reuse_variables()
        input = tf.unpack(tf.transpose(input, [1, 0, 2]))  # L * [N, W]
        embedding = tf.get_variable('embedding', [vocab_size, embed_dim])

        embedded = []
        for word in input:
            embed = tf.nn.embedding_lookup(embedding, word) # [N, W, d]
            conved = TDNN(embed, embed_dim)

            if use_batch_norm:
                conved = batch_norm(conved)

            if use_highway:
                conved = highway(conved, conved.get_shape()[1], highway_layers, 0)

            embedded.append(conved)
            scope.reuse_variables()

    return embedded


def batch_norm(x, epsilon=1e-5):
    shape = x.get_shape().as_list()

    with tf.variable_scope('BatchNorm'):
        gamma = tf.get_variable("gamma", [shape[-1]],
            initializer=tf.random_normal_initializer(1., 0.02))
        beta = tf.get_variable("beta", [shape[-1]],
            initializer=tf.constant_initializer(0.))


        mean, variance = tf.nn.moments(x, [0, 1])

        return tf.nn.batch_norm_with_global_normalization(
            x, mean, variance, beta, gamma, epsilon,
            scale_after_normalization=True)


def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    with tf.variable_scope('Highway'):
        output = input_
        for idx in range(layer_size):
            output = f(linear(output, size, 0, scope='output_lin_%d' % idx, init='he'))

            transform_gate = tf.sigmoid(linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
            carry_gate = 1. - transform_gate

            output = transform_gate * output + carry_gate * input_

    return output
""" Time-delayed Neural Network (cf. http://arxiv.org/abs/1508.06615v4)
    Codes from https://github.com/carpedm20/lstm-char-cnn-tensorflow."""

import tensorflow as tf

def conv2d(input_, output_dim, k_h, k_w,
           stddev=0.02, name="conv2d"):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, 1, 1, 1], padding='VALID')
        return conv

def TDNN(input_, embed_dim=650,
               feature_maps=[5, 10, 15, 20, 20, 20, 20], kernels=[1,2,3,4,5,6,7]):
    """ Apply TDNN.
    Args:
      embed_dim: the dimensionality of the inputs
      feature_maps: list of feature maps (for each kernel width)
      kernels: list of # of kernels (width)
    """
    input_ = tf.expand_dims(input_, -1)  # [batch_size x seq_length x embed_dim x 1]
    layers = []
    for idx, kernel_dim in enumerate(kernels):
        reduced_length = input_.get_shape()[1] - kernel_dim + 1

        # [batch_size x seq_length x embed_dim x feature_map_dim]
        conv = conv2d(input_, feature_maps[idx], kernel_dim , embed_dim, name="kernel%d" % idx)

        # [batch_size x 1 x 1 x feature_map_dim]
        pool = tf.nn.max_pool(tf.tanh(conv), [1, reduced_length, 1, 1], [1, 1, 1, 1], 'VALID')

        layers.append(tf.squeeze(pool, [1, 2]))  # TODO: 배치사이즈 1이라서 이짓함. 원래는 이럴필요없음 - 원본 소스 참고

    return tf.concat(1, layers) if len(kernels) > 1 else layers[0]

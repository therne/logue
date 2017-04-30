import math
import collections
import six
import tensorflow as tf
import numpy as np


def weight(name, shape, init='xavier', range=None):
    """ Initializes weight.
    Args:
      name: Variable name
      shape: Tensor shape
      init: Init mode. xavier / normal / uniform / he (default is 'xavier')
      range: random range if uniform init is used
    """
    initializer = tf.constant_initializer()
    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)

    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev=std)

    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev=0.1)

    elif init == 'uniform':
        if range is None:
            raise ValueError("range must not be None if uniform init is used.")
        initializer = tf.random_uniform_initializer(-range, range)

    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def bias(name, dims):
    return tf.get_variable(name, [dims])


def new_weight(shape, init='xavier'):
    """ Creates weight without variable scope. """
    initial_value = tf.zeros(shape)

    if init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initial_value = tf.random_uniform(shape, -range, range)

    return tf.Variable(initial_value)


def _get_dims(shape):
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    return fan_in, fan_out


def _is_sequence(seq):
  return (isinstance(seq, collections.Sequence)
          and not isinstance(seq, six.string_types))


def linear(args, output_size, bias=True, bias_start=0.0, scope=None, init='xavier'):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
    init: Initialization method.

    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]

    if isinstance(output_size, tf.Dimension):
        output_size = output_size.value

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = weight("Matrix", [total_arg_size, output_size], init)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable("Bias", [output_size],
                initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def flatten(tensor):
    return tf.reshape(tensor, [-1])

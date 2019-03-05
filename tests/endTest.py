import tensorflow as tf
import six
import numpy as np

__INTEGER_TYPES = (
    six.integer_types +
    (np.integer, np.int, np.uint,
     np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64)
)

axis = 1
log_joint = tf.constant([[1,2],[1,2]])
latent_log_prob = tf.constant([[2,3],[2,5]])
sess = tf.Session()

def zs_objective(func, **kwargs):
    """
    Create a :class:`zhusuan.variational.VariationalObjective` with
    pre-computed log-joint, by specified algorithm.

    Args:
        func: The variational algorithm from ZhuSuan. Supported
            functions are: 1. :func:`zhusuan.variational.elbo`
            2. :func:`zhusuan.variational.importance_weighted_objective`
            3. :func:`zhusuan.variational.klpq`
        \\**kwargs: Named arguments passed to `func`.

    Returns:
        zhusuan.variational.VariationalObjective: The constructed
            per-data variational objective.
    """
    return func(
        log_joint=lambda observed: log_joint,
        observed={},
        latent={i: (None, log_prob)
                for i, log_prob in enumerate(latent_log_prob)},
        axis=axis,
        **kwargs
    )

def zs_importance_weighted_objective():
    """
    Create a :class:`zhusuan.variational.ImportanceWeightedObjective`,
    with pre-computed log-joint.

    Returns:
        zhusuan.variational.ImportanceWeightedObjective: The constructed
            per-data importance weighted objective.
    """
    import zhusuan as zs
    return zs_objective(zs.variational.importance_weighted_objective)


def vimco(name=None):
    """
    Get the VIMCO training objective.

    Returns:
        tf.Tensor: The per-data VIMCO training objective.

    See Also:
        :meth:`zhusuan.variational.ImportanceWeightedObjective.vimco`
    """
    with tf.name_scope(name, default_name='vimco'):
        print("vimco by zhusuan")
        print(sess.run(zs_importance_weighted_objective().vimco()))
        # return zs_importance_weighted_objective().vimco()

def vimco_estimator(values, latent_log_prob, axis=None, keepdims=False, name=None):
    """
    my vimco copy from zhusuan
    do not know whether it is right

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        latent_log_prob: The log-densities
                of latent variables from the variational net.
        axis: The sampling axes to be reduced in outputs.
            If not specified, no axis will be reduced.
        keepdims (bool): When `axis` is specified, whether or not to keep
            the reduced axes?  (default :obj:`False`)

    Returns:
        tf.Tensor: The surrogate for optimizing the original target.
            Maximizing/minimizing this surrogate via gradient descent will
            effectively maximize/minimize the original target.

    """

    l_signal = values

    # check size along the sample axis
    err_msg = "VIMCO is a multi-sample gradient estimator, size along " \
              "`axis` in the objective should be larger than 1."
    if l_signal.get_shape()[axis:axis + 1].is_fully_defined():
        if l_signal.get_shape()[axis].value < 2:
            raise ValueError(err_msg)
    _assert_size_along_axis = tf.assert_greater_equal(
        tf.shape(l_signal)[axis], 2, message=err_msg)
    with tf.control_dependencies([_assert_size_along_axis]):
        l_signal = tf.identity(l_signal)

    # compute variance reduction term
    mean_except_signal = (
                                 tf.reduce_sum(l_signal, axis, keepdims=True) - l_signal
                         ) / tf.to_float(tf.shape(l_signal)[axis] - 1)
    x, sub_x = tf.to_float(l_signal), tf.to_float(mean_except_signal)

    n_dim = tf.rank(x)
    axis_dim_mask = tf.cast(tf.one_hot(axis, n_dim), tf.bool)
    original_mask = tf.cast(tf.one_hot(n_dim - 1, n_dim), tf.bool)
    axis_dim = tf.ones([n_dim], tf.int32) * axis
    originals = tf.ones([n_dim], tf.int32) * (n_dim - 1)
    perm = tf.where(original_mask, axis_dim, tf.range(n_dim))
    perm = tf.where(axis_dim_mask, originals, perm)
    multiples = tf.concat(
        [tf.ones([n_dim], tf.int32), [tf.shape(x)[axis]]], 0)

    x = tf.transpose(x, perm=perm)
    sub_x = tf.transpose(sub_x, perm=perm)
    x_ex = tf.tile(tf.expand_dims(x, n_dim), multiples)
    x_ex = x_ex - tf.matrix_diag(x) + tf.matrix_diag(sub_x)
    control_variate = tf.transpose(log_mean_exp(x_ex, n_dim - 1),
                                   perm=perm)

    # variance reduced objective
    l_signal = log_mean_exp(l_signal, axis,
                            keepdims=True) - control_variate
    fake_term = tf.reduce_sum(
        latent_log_prob * tf.stop_gradient(l_signal), axis)
    cost = -fake_term - log_mean_exp(values, axis)

    return cost

def my_vimco():
    """
        Get the VIMCO training objective.

        Returns:
            tf.Tensor: The per-data VIMCO training objective.

        See Also:
            :meth:`zhusuan.variational.ImportanceWeightedObjective.vimco`
    """
    with tf.name_scope(name, default_name='my_vimco'):
        print(sess.run(vimco_estimator(
            values=log_joint - latent_log_prob,
            latent_log_prob=latent_log_prob,
            axis=axis
        )))
def log_mean_exp(x, axis=None, keepdims=False, name=None):
    """
    Compute :math:`\\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)`.

    .. math::

        \\begin{align*}
            \\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max}) \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}

    Args:
        x (Tensor): The input `x`.
        axis (int or tuple[int]): The dimension to take average.
            Default :obj:`None`, all dimensions.
        keepdims (bool): Whether or not to keep the summed dimensions?
            (default :obj:`False`)

    Returns:
        tf.Tensor: The computed value.
    """
    axis = validate_int_tuple_arg('axis', axis, nullable=True) # 把它变成元组
    x = tf.convert_to_tensor(x)
    with tf.name_scope(name, default_name='log_mean_exp', values=[x]):
        x = tf.convert_to_tensor(x)
        x_max_keepdims = tf.reduce_max(x, axis=axis, keepdims=True)
        if not keepdims:
            x_max = tf.squeeze(x_max_keepdims, axis=axis) # 删除张量中维度为1的
        else:
            x_max = x_max_keepdims
        mean_exp = tf.reduce_mean(tf.exp(x - x_max_keepdims), axis=axis,
                                  keepdims=keepdims)
        return x_max + tf.log(mean_exp)

def validate_int_tuple_arg(arg_name, arg_value, nullable=False):
    """
    Validate an integer or a tuple of integers, as a tuple of integers.

    Args:
        arg_name (str): Name of the argument.
        arg_value (int or Iterable[int]): An integer, or an iterable collection
            of integers, to be casted into tuples of integers.
        nullable (bool): Whether or not :obj:`None` value is accepted?

    Returns:
        tuple[int]: The tuple of integers.
    """
    if arg_value is None and nullable:
        pass
    elif is_integer(arg_value):
        arg_value = (arg_value,)
    else:
        try:
            arg_value = tuple(int(v) for v in arg_value)
        except (ValueError, TypeError):
            raise ValueError('Invalid value for argument `{}`: expected to be '
                             'a tuple of integers, but got {!r}.'.
                             format(arg_name, arg_value))
    return arg_value

vimco()
my_vimco()
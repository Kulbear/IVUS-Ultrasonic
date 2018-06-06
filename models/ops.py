import tensorflow as tf
from tensorflow import layers as L

he_init = tf.keras.initializers.he_normal()


def batch_norm(inputs, is_training, data_format='channels_first'):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return L.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=0.95, epsilon=1e-5, center=True,
      scale=True, training=is_training, fused=True)

# def batch_norm(input,
#                is_training,
#                momentum=0.9,
#                epsilon=1e-5,
#                in_place_update=True,
#                data_format='NCHW',
#                scope=None):
#     if in_place_update:
#         return tf.contrib.layers.batch_norm(
#             input,
#             decay=momentum,
#             center=True,
#             scale=True,
#             epsilon=epsilon,
#             fused=True,
#             data_format=data_format,
#             updates_collections=None,
#             is_training=is_training,
#             scope=scope)
#     else:
#         return tf.contrib.layers.batch_norm(
#             input,
#             decay=momentum,
#             center=True,
#             scale=True,
#             fused=True,
#             epsilon=epsilon,
#             data_format=data_format,
#             is_training=is_training,
#             scope=scope)


def main_branch(net,
                ref,
                depth,
                pooling,
                activation,
                is_training,
                lyr_name,
                data_format='channels_first',
                init=he_init):

    net_ = L.conv2d(
        net,
        depth, [5, 5],
        strides=1,
        padding='SAME',
        kernel_initializer=he_init,
        data_format=data_format,
        name='{}_Main_W1'.format(lyr_name))
    net_ = activation(net_, name='{}_Main_A1'.format(lyr_name))
    net_ = batch_norm(net_, is_training)
    net_ = L.conv2d(
        net_,
        depth, [5, 5],
        strides=1,
        padding='SAME',
        kernel_initializer=he_init,
        data_format=data_format,
        name='{}_Main_W2'.format(lyr_name))
    net_ = batch_norm(net_, is_training)
    net_ = tf.add(net_, ref, name='{}_RefMainSum'.format(lyr_name))
    net_ = activation(net_, name='{}_Main_A2'.format(lyr_name))

    return net_


def refining_branch(net,
                    depth,
                    pooling,
                    activation,
                    is_training,
                    lyr_name,
                    data_format='channels_first',
                    init=he_init):
    net_ = L.conv2d(
        net,
        depth, [3, 3],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        data_format=data_format,
        name='{}_Ref_W1'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A1'.format(lyr_name))
    net_ = batch_norm(net_, is_training)
    net_ = L.conv2d(
        net_,
        depth, [1, 1],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        data_format=data_format,
        name='{}_Ref_W2'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A2'.format(lyr_name))
    net_ = batch_norm(net_, is_training)
    net_ = L.conv2d(
        net_,
        depth, [3, 3],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        data_format=data_format,
        name='{}_Ref_W3'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A3'.format(lyr_name))
    net_ = batch_norm(net_, is_training)
    net_ = L.conv2d(
        net_,
        depth, [1, 1],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        data_format=data_format,
        name='{}_Ref_W4'.format(lyr_name))
    net_ = batch_norm(net_, is_training)

    return net_


def downsampling_branch(net,
                        depth,
                        pooling,
                        activation,
                        is_training,
                        lyr_name,
                        data_format='NCHW',
                        init=he_init):
    """
          /==>conv2d(K3S2)==>A==>BN==>\
    net==>                             ==>concat==>out
          \==>pooling(K2S2)==========>/
    """
    _data_format = 'channels_first' if data_format == 'NCHW' else 'channels_last'
    ksize = [1, 1, 2, 2] if data_format == 'NCHW' else [1, 2, 2, 1]
    strides = [1, 1, 2, 2] if data_format == 'NCHW' else [1, 2, 2, 1]
    net_pool = pooling(
        net,
        ksize=ksize,
        strides=strides,
        padding='SAME',
        data_format=data_format,
        name='{}_Pool'.format(lyr_name))
    net_conv = L.conv2d(
        net,
        depth, [3, 3],
        strides=2,
        padding='SAME',
        data_format=_data_format,
        kernel_initializer=init,
        name='{}_PoolW'.format(lyr_name))
    net_conv = activation(net_conv, name='{}_PoolW_A'.format(lyr_name))
    net_conv = batch_norm(net_conv, is_training)
    net_ = tf.concat(axis=1 if data_format == 'NCHW' else -1, values=[net_pool, net_conv])

    return net_


def upsampling_branch(net,
                      depth,
                      activation,
                      is_training,
                      lyr_name,
                      data_format='channels_first',
                      init=he_init):
    """
    net==>deconv2d(K2S2)==>A==>BN==>out
    """
    net_ = L.conv2d_transpose(
        net,
        depth, [2, 2],
        strides=2,
        padding='SAME',
        data_format=data_format,
        kernel_initializer=init,
        name='{}_USampW'.format(lyr_name))
    net_ = activation(net_, name='{}_USampA'.format(lyr_name))
    net_ = batch_norm(net_, is_training)

    return net_


def restoring_branch(net,
                     depth,
                     activation,
                     is_training,
                     init=he_init,
                     data_format='channels_first',
                     name=''):
    net_ = L.conv2d_transpose(
        net,
        depth, [2, 2],
        strides=2,
        padding='SAME',
        data_format=data_format,
        kernel_initializer=init,
        name='{}_W1'.format(name))
    net_ = activation(net_, name='{}_A1'.format(name))
    net_ = batch_norm(net_, is_training)
    net_ = L.conv2d(
        net_,
        depth, [3, 3],
        strides=1,
        padding='SAME',
        data_format=data_format,
        kernel_initializer=init,
        name='{}_W2'.format(name))
    net_ = activation(net_, name='{}_A2'.format(name))
    net_ = batch_norm(net_, is_training)
    return net_

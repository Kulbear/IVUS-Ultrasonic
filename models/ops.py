import tensorflow as tf
from tensorflow import layers as L

he_init = tf.keras.initializers.he_normal()


def batch_norm(input,
               is_training,
               momentum=0.9,
               epsilon=1e-5,
               in_place_update=True,
               scope=None):
    if in_place_update:
        return tf.contrib.layers.batch_norm(
            input,
            decay=momentum,
            center=True,
            scale=True,
            epsilon=epsilon,
            updates_collections=None,
            is_training=is_training,
            scope=scope)
    else:
        return tf.contrib.layers.batch_norm(
            input,
            decay=momentum,
            center=True,
            scale=True,
            epsilon=epsilon,
            is_training=is_training,
            scope=scope)


def main_branch(net,
                ref,
                depth,
                pooling,
                activation,
                is_training,
                lyr_name,
                init=he_init):
    net_ = L.conv2d(
        net,
        depth, [5, 5],
        strides=1,
        padding='SAME',
        kernel_initializer=he_init,
        name='{}_Main_W1'.format(lyr_name))
    net_ = activation(net_, name='{}_Main_A1'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Main_BN1'.format(lyr_name))
    net_ = L.conv2d(
        net_,
        depth, [5, 5],
        strides=1,
        padding='SAME',
        kernel_initializer=he_init,
        name='{}_Main_W2'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Main_BN2'.format(lyr_name))
    net_ = tf.add(net_, ref, name='{}_RefMainSum'.format(lyr_name))
    net_ = activation(net_, name='{}_Main_A2'.format(lyr_name))

    return net_


def refining_branch(net,
                    depth,
                    pooling,
                    activation,
                    is_training,
                    lyr_name,
                    init=he_init):
    net_ = L.conv2d(
        net,
        depth, [3, 3],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        name='{}_Ref_W1'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A1'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Ref_BN1'.format(lyr_name))
    net_ = L.conv2d(
        net_,
        depth, [1, 1],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        name='{}_Ref_W2'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A2'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Ref_BN2'.format(lyr_name))
    net_ = L.conv2d(
        net_,
        depth, [3, 3],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        name='{}_Ref_W3'.format(lyr_name))
    net_ = activation(net_, name='{}_Ref_A3'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Ref_BN3'.format(lyr_name))
    net_ = L.conv2d(
        net_,
        depth, [1, 1],
        strides=1,
        padding='SAME',
        kernel_initializer=init,
        name='{}_Ref_W4'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_Ref_BN4'.format(lyr_name))

    return net_


def downsampling_branch(net,
                        depth,
                        pooling,
                        activation,
                        is_training,
                        lyr_name,
                        init=he_init):
    """
          /==>conv2d(K3S2)==>A==>BN==>\
    net==>                             ==>concat==>out
          \==>pooling(K2S2)==========>/
    """
    net_pool = pooling(
        net,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name='{}_Pool'.format(lyr_name))
    net_conv = L.conv2d(
        net,
        depth, [3, 3],
        strides=2,
        padding='SAME',
        kernel_initializer=init,
        name='{}_PoolW'.format(lyr_name))
    net_conv = activation(net_conv, name='{}_PoolW_A'.format(lyr_name))
    net_conv = batch_norm(
        net_conv, is_training, scope='{}_PoolW_BN'.format(lyr_name))
    net_ = tf.concat(axis=-1, values=[net_pool, net_conv])

    return net_


def upsampling_branch(net,
                      depth,
                      pooling,
                      activation,
                      is_training,
                      lyr_name,
                      init=he_init):
    """
    net==>deconv2d(K2S2)==>A==>BN==>out
    """
    net_ = L.conv2d_transpose(
        net,
        depth, [2, 2],
        strides=2,
        padding='SAME',
        kernel_initializer=init,
        name='{}_USampW'.format(lyr_name))
    net_ = activation(net_, name='{}_USampA'.format(lyr_name))
    net_ = batch_norm(net_, is_training, scope='{}_USampBN'.format(lyr_name))

    return net_


def restoring_branch(net,
                     depth,
                     activation,
                     is_training,
                     config={},
                     init=he_init):
    net_ = L.conv2d_transpose(
        net,
        depth, [3, 3],
        strides=2,
        padding='SAME',
        kernel_initializer=he_init,
        name='Res_W1')
    net_ = batch_norm(net_, is_training, scope='Res_BN1')
    net_ = activation(net_, name='Res_A1')
    if config.state_size[0] == 128:
        net_ = L.conv2d_transpose(
            net_,
            depth, [3, 3],
            strides=2,
            padding='SAME',
            kernel_initializer=he_init,
            name='Res_W2')
        net_ = batch_norm(net_, is_training, scope='Res_BN2')
        net_ = activation(net_, name='Res_A2')

    return net_

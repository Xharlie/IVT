import tensorflow as tf
import tf_util

def get_ivt_basic(src_pc, globalfeats, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
                            weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print( 'net', net.shape)
    print( 'globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net, globalfeats_expand])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training, 
        weight_decay=wd, scope='fold2/conv2')

    return net


def get_ivt_basic_imgfeat_onestream(src_pc, globalfeats, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('net', net.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net, globalfeats_expand, point_feat])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')

    return net


def get_ivt_basic_imgfeat_onestream_skip(src_pc, globalfeats, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net1 = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('net', net.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net, globalfeats_expand, point_feat])

    net = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')

    net = tf_util.conv2d(net+net1, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')

    return net


def get_ivt_basic_imgfeat_twostream(src_pc, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None):

    net = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
                            weight_decay=wd, scope='fold1/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv2')
    net = tf_util.conv2d(net, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold1/conv3')

    concat = tf.concat(axis=3, values=[net, point_feat])

    net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv1')
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay, bn=bn, is_training=is_training,
        weight_decay=wd, scope='fold2/conv2')

    return net

def xyz_ivthead(net, batch_size, wd=None):

    pred = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/xyz_conv')

    pred = tf.reshape(pred, [batch_size, -1, 3])

    return pred

def dist_direct_ivthead(net, batch_size, wd=None):

    direction = tf_util.conv2d(net, 3, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/direction_conv')

    norm = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(direction), axis=3, keepdims=True)), 1e-6)
    print("norm.get_shape().as_list(): ", norm.get_shape().as_list())
    direction = tf.reshape(direction/norm, [batch_size, -1, 3])

    pred_dist = tf.math.abs(tf_util.conv2d(net, 1, [1,1], padding='VALID', stride=[1,1], activation_fn=None, bn=False, weight_decay=wd, scope='head/dist_conv'))

    pred_dist = tf.reshape(pred_dist, [batch_size, -1, 1])

    return pred_dist, direction
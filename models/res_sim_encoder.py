import tensorflow as tf
import tf_util
from tf_util import ResidueBlock

def get_ivt_basic_imgfeat_onestream_skip(src_pc, globalfeats, point_feat, is_training, batch_size, num_point, bn, bn_decay, wd=None, activation_fn=tf.nn.relu):

    net1 = tf_util.conv2d(tf.expand_dims(src_pc,2), 64, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv1')
    net2 = tf_util.conv2d(net1, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv2')
    net3 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold1/conv3')

    globalfeats = tf.reshape(globalfeats, [batch_size, 1, 1, -1])
    globalfeats_expand = tf.tile(globalfeats, [1, src_pc.get_shape()[1], 1, 1])
    print('skip net3', net3.shape)
    print('globalfeats_expand', globalfeats_expand.shape)
    concat = tf.concat(axis=3, values=[net3+net2, globalfeats_expand, point_feat])

    net4 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv1')

    net5 = tf_util.conv2d(net4+net3, 256, [1,1], padding='VALID', stride=[1,1], activation_fn=activation_fn, bn_decay=bn_decay, bn=bn, is_training=is_training, weight_decay=wd, scope='fold2/conv2')

    return net5


def res_sim_encoder(inputs, batch_size, is_training = True, activation_fn=tf.nn.relu, bn=True, bn_decay=None, wd=None):
    encdr_end_points=[]
    net = ResidueBlock(inputs, is_training, 3, 32, "res_block1", 1, 2, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd) #112
    net = ResidueBlock(net, is_training, 32, 128, "res_block2", 1, 2, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd) #56
    encdr_end_points.append(net)
    net = ResidueBlock(net, is_training, 128, 256, "res_block3", 1, 2, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd) #28
    encdr_end_points.append(net)
    net = ResidueBlock(net, is_training, 256, 256, "res_block4", 1, 2, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd) #14
    encdr_end_points.append(net)
    net = ResidueBlock(net, is_training, 256, 256, "res_block5", 1, 2, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd)  # 7
    encdr_end_points.append(net)
    net = ResidueBlock(net, is_training, 256, 64, "res_block6", 1, 1, activation_fn=activation_fn, residue_ratio=0.5, use_BatchStatistics=False, wd=wd)  # 7
    net = tf_util.fully_connected(tf.reshape(net,[batch_size, -1]), 1024, "res_fc", weight_decay=wd, activation_fn=activation_fn, bn=bn, bn_decay=bn_decay, is_training=is_training)
    return net, encdr_end_points
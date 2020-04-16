import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg, resnet_v1, resnet_v2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'data'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
print(os.path.join(BASE_DIR, 'models'))
import ivtnet

def placeholder_inputs(scope='', FLAGS=None, num_pnts=None):
    if num_pnts is None:
        num_pnts = FLAGS.num_pnts
    with tf.compat.v1.variable_scope(scope) as sc:
        pnts_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, num_pnts, 3))
        if FLAGS.alpha:
            imgs_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_h, FLAGS.img_w, 4))
        else:
            imgs_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_h, FLAGS.img_w, 3))
        ivts_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, num_pnts, 3))
        obj_rot_mat_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, 3, 3))
        trans_mat_pl = tf.compat.v1.placeholder(tf.float32, shape=(FLAGS.batch_size, 4, 3))
    ivt = {}
    ivt['pnts'] = pnts_pl
    ivt['ivts'] = ivts_pl
    ivt['imgs'] = imgs_pl
    ivt['obj_rot_mats'] = obj_rot_mat_pl
    ivt['trans_mats'] = trans_mat_pl
    return ivt


# def placeholder_features(batch_size, num_sample_pc = 256, scope=''):
#     with tf.compat.v1.variable_scope(scope) as sc:
#         ref_feats_embedding_cnn_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1, 1, 1024))
#         point_img_feat_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_sample_pc, 1, 1472))
#     feat = {}
#     feat['ref_feats_embedding_cnn'] = ref_feats_embedding_cnn_pl
#     feat['point_img_feat'] = point_img_feat_pl
#     return feat

def get_model(input_pls, is_training, bn=False, bn_decay=None, img_size = 224, FLAGS=None):

    input_imgs = input_pls['imgs']
    input_pnts = input_pls['pnts']
    input_ivts = input_pls['ivts']
    input_trans_mat = input_pls['trans_mats']
    input_obj_rot_mats = input_pls['obj_rot_mats']

    batch_size = input_imgs.get_shape()[0].value

    # endpoints
    end_points = {}
    end_points['pnts'] = input_pnts
    if FLAGS.rot:
        end_points['gt_ivts_xyz'] = tf.matmul(input_ivts, input_obj_rot_mats)
        end_points['pnts_rot'] = tf.matmul(input_pnts, input_obj_rot_mats)
    else:
        end_points['gt_ivts_xyz'] = input_ivts #* 10
        end_points['pnts_rot'] = input_pnts
    input_pnts_rot = end_points['pnts_rot']
    end_points['imgs'] = input_imgs # B*H*W*3|4

    # Image extract features
    if input_imgs.shape[1] != img_size or input_imgs.shape[2] != img_size:
        if FLAGS.alpha:
            ref_img_rgb = tf.compat.v1.image.resize_bilinear(input_imgs[:,:,:,:3], [img_size, img_size])
            ref_img_alpha = tf.image.resize_nearest_neighbor(
                tf.expand_dims(input_imgs[:,:,:,3], axis=-1), [img_size, img_size])
            ref_img = tf.concat([ref_img_rgb, ref_img_alpha], axis = -1)
        else:
            ref_img = tf.compat.v1.image.resize_bilinear(input_imgs, [img_size, img_size])
    else:
        ref_img = input_imgs
    end_points['resized_ref_img'] = ref_img
    if FLAGS.encoder == "vgg_16":
        vgg.vgg_16.default_image_size = img_size
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(FLAGS.wd)):
            ref_feats_embedding, encdr_end_points = vgg.vgg_16(ref_img, num_classes=FLAGS.num_classes, is_training=False, scope='vgg_16', spatial_squeeze=False)
    elif FLAGS.encoder == "resnet_v1_50":
        resnet_v1.default_image_size = img_size
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            ref_feats_embedding, encdr_end_points = resnet_v1.resnet_v1_50(ref_img, FLAGS.num_classes, is_training=is_training, scope='resnet_v1_50')
        scopelst = ["resnet_v1_50/block1","resnet_v1_50/block2","resnet_v1_50/block3",'resnet_v1_50/block4']
    elif FLAGS.encoder == "resnet_v1_101":
        resnet_v1.default_image_size = img_size
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            ref_feats_embedding, encdr_end_points = resnet_v1.resnet_v1_101(ref_img, FLAGS.num_classes, is_training=is_training, scope='resnet_v1_101')
        scopelst = ["resnet_v1_101/block1", "resnet_v1_101/block2", "resnet_v1_101/block3", 'resnet_v1_101/block4']
    elif FLAGS.encoder == "resnet_v2_50":
        resnet_v2.default_image_size = img_size
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            ref_feats_embedding, encdr_end_points = resnet_v2.resnet_v2_50(ref_img, FLAGS.num_classes, is_training=is_training, scope='resnet_v2_50')
        scopelst = ["resnet_v2_50/block1", "resnet_v2_50/block2", "resnet_v2_50/block3", 'resnet_v2_50/block4']
    elif FLAGS.encoder == "resnet_v2_101":
        resnet_v2.default_image_size = img_size
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            ref_feats_embedding, encdr_end_points = resnet_v2.resnet_v2_101(ref_img, FLAGS.num_classes, is_training=is_training, scope='resnet_v2_101')
        scopelst = ["resnet_v2_101/block1", "resnet_v2_101/block2", "resnet_v2_101/block3", 'resnet_v2_101/block4']
    ref_feats_embedding_cnn = tf.squeeze(ref_feats_embedding, axis=[1, 2])
    end_points['img_embedding'] = ref_feats_embedding_cnn
    point_img_feat=None
    ivts_feat=None
    sample_img_points = get_img_points(input_pnts, input_trans_mat)  # B * N * 2


    if FLAGS.img_feat_onestream:
        with tf.compat.v1.variable_scope("sdfimgfeat") as scope:
            if FLAGS.encoder[:3] == "vgg":
                conv1 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
                point_conv1 = tf.contrib.resampler.resampler(conv1, sample_img_points)
                conv2 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
                point_conv2 = tf.contrib.resampler.resampler(conv2, sample_img_points)
                conv3 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
                point_conv3 = tf.contrib.resampler.resampler(conv3, sample_img_points)
                conv4 = tf.compat.v1.image.resize_bilinear(encdr_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
                point_conv4 = tf.contrib.resampler.resampler(conv4, sample_img_points)
                point_img_feat = tf.concat(axis=2, values=[point_conv1, point_conv2, point_conv3, point_conv4]) # small
            elif FLAGS.encoder[:3] == "res":
                # print(encdr_end_points.keys())
                conv1 = tf.compat.v1.image.resize_bilinear(encdr_end_points[scopelst[0]], (FLAGS.img_h, FLAGS.img_w))
                point_conv1 = tf.contrib.resampler.resampler(conv1, sample_img_points)
                conv2 = tf.compat.v1.image.resize_bilinear(encdr_end_points[scopelst[1]], (FLAGS.img_h, FLAGS.img_w))
                point_conv2 = tf.contrib.resampler.resampler(conv2, sample_img_points)
                conv3 = tf.compat.v1.image.resize_bilinear(encdr_end_points[scopelst[2]], (FLAGS.img_h, FLAGS.img_w))
                point_conv3 = tf.contrib.resampler.resampler(conv3, sample_img_points)
                # conv4 = tf.compat.v1.image.resize_bilinear(encdr_end_points[scopelst[3]], (FLAGS.img_h, FLAGS.img_w))
                # point_conv4 = tf.contrib.resampler.resampler(conv4, sample_img_points)
                point_img_feat = tf.concat(axis=2, values=[point_conv1, point_conv2, point_conv3])
            print("point_img_feat.shape", point_img_feat.get_shape())
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            ivts_feat = ivtnet.get_ivt_basic_imgfeat_onestream(input_pnts_rot, ref_feats_embedding_cnn, point_img_feat, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay, wd=FLAGS.wd)
    else:
        if not FLAGS.multi_view:
            with tf.compat.v1.variable_scope("sdfprediction") as scope:
                ivts_feat = ivtnet.get_ivt_basic(input_pnts_rot, ref_feats_embedding_cnn, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay,wd=FLAGS.wd)
    end_points['pred_ivts_xyz'], end_points['pred_ivts_dist'], end_points['pred_ivts_direction'] = None, None, None
    if FLAGS.XYZ:
        end_points['pred_ivts_xyz'] = ivtnet.xyz_ivthead(ivts_feat, batch_size, wd=FLAGS.wd)
        end_points['pred_ivts_dist'] = tf.sqrt(tf.reduce_sum(tf.square(end_points['pred_ivts_xyz']), axis=2, keepdims=True))
        end_points['pred_ivts_direction'] = end_points['pred_ivts_xyz'] / tf.maximum(end_points['pred_ivts_dist'], 1e-6)
    else:
        end_points['pred_ivts_dist'], end_points['pred_ivts_direction'] = ivtnet.dist_direct_ivthead(ivts_feat, batch_size, wd=FLAGS.wd)
        end_points['pred_ivts_xyz'] = end_points['pred_ivts_direction'] * end_points['pred_ivts_dist']

    end_points["sample_img_points"] = sample_img_points
    end_points["ref_feats_embedding_cnn"] = ref_feats_embedding_cnn
    end_points["point_img_feat"] = point_img_feat

    return end_points

def get_img_points(sample_pc, trans_mat_right):
    # sample_pc B*N*3
    size_lst = sample_pc.get_shape().as_list()
    homo_pc = tf.concat((sample_pc, tf.ones((size_lst[0], size_lst[1], 1),dtype=np.float32)),axis= -1)
    print("homo_pc.get_shape()", homo_pc.get_shape())
    pc_xyz = tf.matmul(homo_pc, trans_mat_right)
    print("pc_xyz.get_shape()", pc_xyz.get_shape()) # B * N * 3
    pc_xy = tf.divide(pc_xyz[:,:,:2], tf.expand_dims(pc_xyz[:,:,2], axis = 2))
    mintensor = tf.constant([0.0,0.0], dtype=tf.float32)
    maxtensor = tf.constant([136.0,136.0], dtype=tf.float32)
    return tf.minimum(maxtensor, tf.maximum(mintensor, pc_xy))


def get_loss(end_points, regularization=True, FLAGS=None):

    gt_ivts_xyz = end_points['gt_ivts_xyz']
    gt_ivts_dist = tf.sqrt(tf.reduce_sum(tf.square(gt_ivts_xyz), axis=2, keepdims=True))
    print("gt_ivts_dist.get_shape().as_list(): ", gt_ivts_dist.get_shape().as_list())
    gt_ivts_direction = gt_ivts_xyz / tf.maximum(gt_ivts_dist, 1e-6)
    end_points['gt_ivts_dist'] = gt_ivts_dist
    end_points['gt_ivts_direction'] = gt_ivts_direction

    if FLAGS.weight_type == 'propor':
        weight_mask = 1 / tf.maximum(gt_ivts_dist, 1e-6)
    elif FLAGS.weight_type == 'ntanh':
        thresh = tf.constant(0.05, dtype=tf.float32)
        weight_mask = tf.cast(tf.less_equal(gt_ivts_dist, thresh),dtype=tf.float32) \
              + tf.cast(tf.greater(gt_ivts_dist, thresh),dtype=tf.float32) * (tf.tanh(thresh - gt_ivts_dist) + 1)
    else:
        weight_mask = tf.ones([FLAGS.batch_size, FLAGS.num_pnts, 1], dtype=tf.float32)
    end_points['weighed_mask'] = weight_mask

    ###########ivt
    print('gt_ivts_xyz, gt_dist, gt_direction, weight_mask shapes:', gt_ivts_xyz.get_shape().as_list(), gt_ivts_dist.get_shape().as_list(), gt_ivts_direction.get_shape().as_list(), weight_mask.get_shape().as_list())
    ################
    # Compute loss #
    ################
    end_points['losses'] = {}

    ############### accuracy
    ivts_xyz_diff = tf.abs(gt_ivts_xyz - end_points['pred_ivts_xyz'])
    ivts_xyz_avg_diff = tf.reduce_mean(ivts_xyz_diff)

    ivts_locnorm_diff = tf.norm(ivts_xyz_diff, ord='euclidean', axis=-1, keepdims=True)
    ivts_locnorm_avg_diff = tf.reduce_mean(ivts_locnorm_diff)

    # ivts_locsqrnorm_diff = tf.square(ivts_locnorm_diff)
    # ivts_locsqrnorm_avg_diff = tf.reduce_mean(ivts_locsqrnorm_diff)

    ivts_locsqrnorm_diff = tf.reduce_sum(tf.square(ivts_xyz_diff), axis=2)
    ivts_locsqrnorm_avg_diff = tf.reduce_mean(ivts_locsqrnorm_diff)

    ivts_dist_diff = tf.abs(gt_ivts_dist - end_points['pred_ivts_dist'])
    ivts_dist_avg_diff = tf.reduce_mean(ivts_dist_diff)
    ivts_direction_diff = tf.reduce_sum(tf.multiply(gt_ivts_direction, end_points['pred_ivts_direction']), axis=2, keepdims=True)
    ivts_direction_avg_diff = tf.reduce_mean(ivts_direction_diff)
    ivts_direction_abs_diff = tf.abs(ivts_direction_diff)
    ivts_direction_abs_avg_diff = tf.reduce_mean(ivts_direction_abs_diff)
    end_points['lvl'] = {}
    if FLAGS.distlimit is not None:
        end_points['lvl']["xyz_lvl_diff"], end_points['lvl']["locnorm_lvl_diff"], end_points['lvl']["locsqrnorm_lvl_diff"], end_points['lvl']["dist_lvl_diff"], end_points['lvl']["direction_lvl_diff"], end_points['lvl']["direction_abs_lvl_diff"], end_points['lvl']["num"] \
            = gen_lvl_diff(ivts_xyz_diff, ivts_locnorm_avg_diff, ivts_locsqrnorm_avg_diff, ivts_dist_diff, ivts_direction_diff, ivts_direction_abs_diff, FLAGS.distlimit, gt_ivts_dist)

    ivts_xyz_loss, ivts_dist_loss, ivts_direction_loss, ivts_direction_abs_loss, ivts_locnorm_loss, ivts_locsqrnorm_loss = 0., 0., 0., 0., 0., 0.
    
    if FLAGS.lossw[0] !=0:
        if FLAGS.weight_type == "non":
            ivts_xyz_loss = ivts_xyz_avg_diff
        else:
            ivts_xyz_loss = tf.reduce_mean(ivts_xyz_diff * weight_mask)
        end_points['losses']['ivts_xyz_loss'] = ivts_xyz_loss
    if FLAGS.lossw[1] != 0:
        if FLAGS.weight_type == "non":
            ivts_locnorm_loss = ivts_locnorm_avg_diff
        else:
            ivts_locnorm_loss = tf.reduce_mean(ivts_locnorm_diff * weight_mask)
        end_points['losses']['ivts_locnorm_loss'] = ivts_locnorm_loss
    if FLAGS.lossw[2] != 0:
        if FLAGS.weight_type == "non":
            ivts_locsqrnorm_loss = ivts_locsqrnorm_avg_diff
        else:
            ivts_locsqrnorm_loss = tf.reduce_mean(ivts_locsqrnorm_diff * weight_mask)
        end_points['losses']['ivts_locsqrnorm_loss'] = ivts_locsqrnorm_loss
    if FLAGS.lossw[3] != 0:
        if FLAGS.weight_type == "non":
            ivts_dist_loss = ivts_dist_avg_diff
        else:
            ivts_dist_loss = tf.reduce_mean(ivts_dist_diff * weight_mask)
        end_points['losses']['ivts_dist_loss'] = ivts_dist_loss
    if FLAGS.lossw[4] != 0:
        ivts_direction_loss = tf.reduce_mean((1.0-ivts_direction_diff))
        end_points['losses']['ivts_direction_loss'] = ivts_direction_loss
    if FLAGS.lossw[5] != 0:
        ivts_direction_abs_loss = tf.reduce_mean((1.0-ivts_direction_abs_diff))
        end_points['losses']['ivts_direction_abs_loss'] = ivts_direction_abs_loss


    # print("weight_mask.get_shape().as_list(): ", weight_mask.get_shape().as_list())
    # print("ivts_xyz_diff.get_shape().as_list(): ", ivts_xyz_diff.get_shape().as_list())
    # print("ivts_xyz_avg_diff.get_shape().as_list(): ", ivts_xyz_avg_diff.get_shape().as_list())
    # print("ivts_dist_diff.get_shape().as_list(): ", ivts_dist_diff.get_shape().as_list())
    # print("ivts_dist_avg_diff.get_shape().as_list(): ", ivts_dist_avg_diff.get_shape().as_list())
    # print("ivts_direction_diff.get_shape().as_list(): ", ivts_direction_diff.get_shape().as_list())
    # print("ivts_direction_avg_diff.get_shape().as_list(): ", ivts_direction_diff.get_shape().as_list())

    loss = FLAGS.lossw[0] * ivts_xyz_loss + FLAGS.lossw[1] * ivts_locnorm_loss + FLAGS.lossw[2] * ivts_locsqrnorm_loss + FLAGS.lossw[3] * ivts_dist_loss + FLAGS.lossw[4] * ivts_direction_loss + FLAGS.lossw[5] * ivts_direction_abs_loss
    end_points['losses']['ivts_xyz_avg_diff'] = ivts_xyz_avg_diff
    end_points['losses']['ivts_dist_avg_diff'] = ivts_dist_avg_diff
    end_points['losses']['ivts_direction_avg_diff'] = ivts_direction_avg_diff
    end_points['losses']['ivts_direction_abs_avg_diff'] = ivts_direction_abs_avg_diff
    end_points['losses']['ivts_locnorm_avg_diff'] = ivts_locnorm_avg_diff
    end_points['losses']['ivts_locsqrnorm_avg_diff'] = ivts_locsqrnorm_avg_diff
    end_points['losses']['loss'] = loss
    ############### weight decay
    if regularization:
        vgg_regularization_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())
        decoder_regularization_loss = tf.add_n(tf.compat.v1.get_collection('regularizer'))
        end_points['losses']['regularization'] = (vgg_regularization_loss + decoder_regularization_loss)
        loss += (vgg_regularization_loss + decoder_regularization_loss)
    end_points['losses']['overall_loss'] = loss
    return loss, end_points


def gen_lvl_diff(ivts_xyz_diff, ivts_locnorm_avg_diff, ivts_locsqrnorm_avg_diff, ivts_dist_diff, ivts_direction_diff, ivts_direction_abs_diff, distlimit, gt_ivts_dist):
    xyz_lvl_diff, locnorm_lvl_diff, locsqrnorm_lvl_diff, dist_lvl_diff, direction_lvl_diff, direction_abs_lvl_diff = [],[],[],[],[],[]
    lvl_num = []
    ones = tf.ones_like(gt_ivts_dist, dtype=tf.float32)
    for i in range(len(distlimit)//2):
        upper = distlimit[i*2] * ones
        lower = distlimit[i*2+1] * ones
        floatmask = tf.cast(tf.compat.v1.logical_and(tf.compat.v1.less_equal(gt_ivts_dist, upper), tf.compat.v1.greater(gt_ivts_dist,lower)), dtype=tf.float32)
        print("floatmask.get_shape().as_list(), ivts_locnorm_avg_diff.get_shape().as_list()",floatmask.get_shape().as_list(), ivts_locnorm_avg_diff.get_shape().as_list())
        lvl_num.append(tf.reduce_sum(floatmask))
        xyz_lvl_diff.append(tf.reduce_sum(floatmask*ivts_xyz_diff))
        locnorm_lvl_diff.append(tf.reduce_sum(floatmask*ivts_locnorm_avg_diff))
        locsqrnorm_lvl_diff.append(tf.reduce_sum(floatmask*ivts_locsqrnorm_avg_diff))
        dist_lvl_diff.append(tf.reduce_sum(floatmask*ivts_dist_diff))
        direction_lvl_diff.append(tf.reduce_sum(floatmask*ivts_direction_diff))
        direction_abs_lvl_diff.append(tf.reduce_sum(floatmask*ivts_direction_abs_diff))
    lvl_num = tf.convert_to_tensor(lvl_num)
    xyz_lvl_diff = tf.convert_to_tensor(xyz_lvl_diff)
    locnorm_lvl_diff = tf.convert_to_tensor(locnorm_lvl_diff)
    locsqrnorm_lvl_diff = tf.convert_to_tensor(locsqrnorm_lvl_diff)
    dist_lvl_diff = tf.convert_to_tensor(dist_lvl_diff)
    direction_lvl_diff = tf.convert_to_tensor(direction_lvl_diff)
    direction_abs_lvl_diff = tf.convert_to_tensor(direction_abs_lvl_diff)
    return xyz_lvl_diff, locnorm_lvl_diff, locsqrnorm_lvl_diff, dist_lvl_diff, direction_lvl_diff, direction_abs_lvl_diff, lvl_num
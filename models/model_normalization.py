import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import vgg
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

def get_model(input_pls, is_training, bn=False, bn_decay=None, img_size = 224, wd=1e-5, FLAGS=None):

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
    vgg.vgg_16.default_image_size = img_size
    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(wd)):
        ref_feats_embedding, vgg_end_points = vgg.vgg_16(ref_img, num_classes=FLAGS.num_classes, is_training=False, scope='vgg_16', spatial_squeeze=False)
        ref_feats_embedding_cnn = tf.squeeze(ref_feats_embedding, axis = [1,2])
    end_points['img_embedding'] = ref_feats_embedding_cnn
    point_img_feat=None
    ivts_feat=None
    sample_img_points = get_img_points(input_pnts, input_trans_mat)  # B * N * 2


    if FLAGS.img_feat_onestream:
        with tf.compat.v1.variable_scope("sdfimgfeat") as scope:
            vgg_conv1 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            vgg_conv2 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            vgg_conv3 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            point_img_feat = tf.concat(axis=2, values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3])
            print("point_img_feat.shape", point_img_feat.get_shape())
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            ivts_feat = ivtnet.get_ivt_basic_imgfeat_onestream(input_pnts_rot, ref_feats_embedding_cnn, point_img_feat, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay, wd=wd)

    elif FLAGS.img_feat_twostream:
        with tf.compat.v1.variable_scope("sdfimgtwofeat") as scope:
            vgg_conv1 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv1/conv1_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv1 = tf.contrib.resampler.resampler(vgg_conv1, sample_img_points)
            print('point_vgg_conv1', point_vgg_conv1.shape)
            vgg_conv2 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv2/conv2_2'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv2 = tf.contrib.resampler.resampler(vgg_conv2, sample_img_points)
            print('point_vgg_conv2', point_vgg_conv2.shape)
            vgg_conv3 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv3/conv3_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv3 = tf.contrib.resampler.resampler(vgg_conv3, sample_img_points)
            print('point_vgg_conv3', point_vgg_conv3.shape)
            vgg_conv4 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv4/conv4_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv4 = tf.contrib.resampler.resampler(vgg_conv4, sample_img_points)
            print('point_vgg_conv4', point_vgg_conv4.shape)
            vgg_conv5 = tf.compat.v1.image.resize_bilinear(vgg_end_points['vgg_16/conv5/conv5_3'], (FLAGS.img_h, FLAGS.img_w))
            point_vgg_conv5 = tf.contrib.resampler.resampler(vgg_conv5, sample_img_points)
            print('point_vgg_conv5', point_vgg_conv5.shape)

            point_img_feat = tf.concat(axis=2, values=[point_vgg_conv1, point_vgg_conv2, point_vgg_conv3, point_vgg_conv4,point_vgg_conv5])
            point_img_feat = tf.expand_dims(point_img_feat, axis=2)
            print('point_img_feat', point_img_feat.shape)
            if not FLAGS.multi_view:
                # Predict SDF
                with tf.compat.v1.variable_scope("sdfprediction") as scope:
                    pred_ivts_global_feat = ivtnet.get_ivt_basic(input_pnts_rot, ref_feats_embedding_cnn, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay, wd=wd)

                with tf.compat.v1.variable_scope("sdfprediction_imgfeat") as scope:
                    pred_ivts_local_feat = ivtnet.get_ivt_basic_imgfeat_twostream(input_pnts_rot, point_img_feat, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay, wd=wd)

                ivts_feat = pred_ivts_global_feat + pred_ivts_local_feat
                end_points["pred_ivts_global_feat"] = pred_ivts_global_feat
                end_points["pred_ivts_local_feat"] = pred_ivts_local_feat
    else:
        if not FLAGS.multi_view:
            with tf.compat.v1.variable_scope("sdfprediction") as scope:
                ivts_feat = ivtnet.get_ivt_basic(input_pnts_rot, ref_feats_embedding_cnn, is_training, batch_size, FLAGS.num_pnts, bn, bn_decay,wd=wd)
    end_points['pred_ivts_xyz'], end_points['pred_ivts_dist'], end_points['pred_ivts_direction'] = None, None, None
    if FLAGS.XYZ:
        end_points['pred_ivts_xyz'] = ivtnet.xyz_ivthead(ivts_feat, batch_size, wd=wd)
        end_points['pred_ivts_dist'] = tf.sqrt(tf.reduce_sum(tf.square(end_points['pred_ivts_xyz']), axis=2, keepdims=True))
        end_points['pred_ivts_direction'] = end_points['pred_ivts_xyz'] / tf.maximum(end_points['pred_ivts_dist'], 1e-6)
    else:
        end_points['pred_ivts_dist'], end_points['pred_ivts_direction'] = ivtnet.dist_direct_ivthead(ivts_feat, batch_size, wd=wd)
        end_points['pred_ivts_xyz'] = end_points['pred_ivts_direction'] * end_points['pred_ivts_dist']

    end_points["sample_img_points"] = sample_img_points
    end_points["ref_feats_embedding_cnn"] = ref_feats_embedding_cnn
    end_points["point_img_feat"] = point_img_feat

    return end_points

def get_decoder(num_point, input_pls, feature_pls, bn=False, bn_decay=None,wd=None):
    ref_feats_embedding_cnn = feature_pls["ref_feats_embedding_cnn"]
    point_img_feat = feature_pls["point_img_feat"]
    input_pnts_rot = input_pls['sample_pc_rot']

    with tf.compat.v1.variable_scope("sdfprediction") as scope:
        pred_ivts_value_global = ivtnet.get_ivts_basic(input_pnts_rot, ref_feats_embedding_cnn, False, 1, num_point, bn, bn_decay, wd=wd)

    with tf.compat.v1.variable_scope("sdfprediction_imgfeat") as scope:
        pred_ivts_value_local = ivtnet.get_ivts_basic_imgfeat_twostream(input_pnts_rot, point_img_feat, False, 1, num_point, bn, bn_decay, wd=wd)
    multi_pred_ivt = pred_ivts_value_global + pred_ivts_value_local
    return multi_pred_ivt


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
    ivts_dist_diff = tf.abs(gt_ivts_dist - end_points['pred_ivts_dist'])
    ivts_dist_avg_diff = tf.reduce_mean(ivts_dist_diff)
    ivts_direction_diff = tf.reduce_sum(tf.multiply(gt_ivts_direction, end_points['pred_ivts_direction']), axis=2, keepdims=True)
    ivts_direction_avg_diff = tf.reduce_mean(ivts_direction_diff)
    ivts_direction_abs_diff = tf.abs(ivts_direction_diff)
    ivts_direction_abs_avg_diff = tf.reduce_mean(ivts_direction_abs_diff)

    ivts_xyz_loss, ivts_dist_loss, ivts_direction_loss, ivts_direction_abs_loss = 0., 0., 0., 0.
    
    if FLAGS.lossw[0] !=0:
        ivts_xyz_loss = tf.reduce_mean(ivts_xyz_diff * weight_mask)
        end_points['losses']['ivts_xyz_loss'] = ivts_xyz_loss
    if FLAGS.lossw[1] != 0:
        ivts_dist_loss = tf.reduce_mean(ivts_dist_diff * weight_mask)
        end_points['losses']['ivts_dist_loss'] = ivts_dist_loss
    if FLAGS.lossw[2] != 0:
        ivts_direction_loss = tf.reduce_mean((1.0-ivts_direction_diff))
        end_points['losses']['ivts_direction_loss'] = ivts_direction_loss
    if FLAGS.lossw[3] != 0:
        ivts_direction_abs_loss = tf.reduce_mean((1.0-ivts_direction_abs_diff))
        end_points['losses']['ivts_direction_abs_loss'] = ivts_direction_abs_loss

    print("weight_mask.get_shape().as_list(): ", weight_mask.get_shape().as_list())
    print("ivts_xyz_diff.get_shape().as_list(): ", ivts_xyz_diff.get_shape().as_list())
    print("ivts_xyz_avg_diff.get_shape().as_list(): ", ivts_xyz_avg_diff.get_shape().as_list())
    print("ivts_dist_diff.get_shape().as_list(): ", ivts_dist_diff.get_shape().as_list())
    print("ivts_dist_avg_diff.get_shape().as_list(): ", ivts_dist_avg_diff.get_shape().as_list())
    print("ivts_direction_diff.get_shape().as_list(): ", ivts_direction_diff.get_shape().as_list())
    print("ivts_direction_avg_diff.get_shape().as_list(): ", ivts_direction_diff.get_shape().as_list())

    loss = FLAGS.lossw[0] * ivts_xyz_loss + FLAGS.lossw[1] * ivts_dist_loss + FLAGS.lossw[2] * ivts_direction_loss + FLAGS.lossw[3] * ivts_direction_abs_loss
    end_points['losses']['ivts_xyz_avg_diff'] = ivts_xyz_avg_diff
    end_points['losses']['ivts_dist_avg_diff'] = ivts_dist_avg_diff
    end_points['losses']['ivts_direction_avg_diff'] = ivts_direction_avg_diff
    end_points['losses']['ivts_direction_abs_avg_diff'] = ivts_direction_abs_avg_diff
    end_points['losses']['loss'] = loss
    ############### weight decay
    if regularization:
        vgg_regularization_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())
        decoder_regularization_loss = tf.add_n(tf.compat.v1.get_collection('regularizer'))
        end_points['losses']['regularization'] = (vgg_regularization_loss + decoder_regularization_loss)
        loss += (vgg_regularization_loss + decoder_regularization_loss)
    end_points['losses']['overall_loss'] = loss
    return loss, end_points

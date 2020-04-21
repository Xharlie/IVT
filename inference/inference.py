from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
import os
import cv2
import sys
import time
from tensorflow.contrib.framework.python.framework import checkpoint_utils
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("PID:", os.getpid())
print(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_load'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
import data_ivt_h5_queue  # as data
import output_utils
import create_file_lst
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing'))
from normal_gen import save_norm
import gpu_create_manifold_ivt as ct
import argparse
import pymesh
import trimesh
import pandas as pd
import normal_gen
from sklearn.neighbors import DistanceMetric as dm
from sklearn.neighbors import NearestNeighbors
from random import sample
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

slim = tf.contrib.slim
FLAGS=None
RESULT_PATH = None

LOG_FOUT = None

MAXOUT = None
NUM_INPUT_POINTS = None

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = None
BN_DECAY_CLIP = 0.99


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
        BN_INIT_DECAY,
        batch * FLAGS.batch_size,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


class NoStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):
    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
    # print(vars_in_pretrained_model)
    vars_in_defined_model = []

    for var in tf.trainable_variables():
        if isinstance(prefixs, list):
            for prefix in prefixs:
                if (var.op.name.startswith(prefix)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                    if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                        vars_in_defined_model.append(var)
        else:
            if (var.op.name.startswith(prefixs)) and (var.op.name in vars_in_pretrained_model.keys()) and ('logits' not in var.op.name):
                if (list(var.shape) == vars_in_pretrained_model[var.op.name]):
                    vars_in_defined_model.append(var)
    # print(vars_in_defined_model)
    saver = tf.train.Saver(vars_in_defined_model)
    try:
        saver.restore(sess, LOAD_MODEL_FILE)
        print("Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print("Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print("Fail loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def load_model_strict(sess, saver, restore_model):
    ckptstate = tf.train.get_checkpoint_state(restore_model)
    if ckptstate is not None:
        LOAD_MODEL_FILE = os.path.join(restore_model, os.path.basename(ckptstate.model_checkpoint_path))
        saver.restore(sess, LOAD_MODEL_FILE)
    return sess

def test(raw_dirs, TEST_DATASET):
    log_string(FLAGS.log_dir)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model.placeholder_inputs(scope='inputs_pl', FLAGS=FLAGS, num_pnts=NUM_INPUT_POINTS)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            print(is_training_pl)
            batch = tf.Variable(0, name='batch')

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, is_training_pl, bn=False, FLAGS=FLAGS)
            loss, end_points = model.get_loss(end_points, FLAGS=FLAGS)
            gpu_options = tf.compat.v1.GPUOptions()  # (per_process_gpu_memory_fraction=0.99)
            config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.compat.v1.Session(config=config)

            ##### all
            # Init variables
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # Overall
            saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) if ('lr' not in v.name) and ('batch' not in v.name)])

            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'step': batch,
                   'saver': saver,
                   'end_points': end_points}
            sys.stdout.flush()

            TEST_DATASET.start()
            log_string('**** INFERENCE  ****')
            sys.stdout.flush()
            test_one_epoch(sess, ops, raw_dirs, TEST_DATASET)
            TEST_DATASET.shutdown()
            print("Done!")

def test_one_epoch(sess, ops, raw_dirs, TEST_DATASET):
    num_batches = int(len(TEST_DATASET) / FLAGS.batch_size)
    print('num_batches', num_batches)
    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = TEST_DATASET.fetch()
        view_id = batch_data["view_id"][0]
        outdir = os.path.join(FLAGS.outdir, batch_data["cat_id"][0], batch_data["obj_nm"][0], str(view_id), FLAGS.unitype)
        os.makedirs(outdir, exist_ok=True)
        nums = FLAGS.initnums
        weightform = FLAGS.weightform
        distr = FLAGS.distr

        sess = load_model_strict(sess,  ops["saver"], FLAGS.restore_model)

        if FLAGS.unitype == "uni":
            loc, norm, std, weights, gt_pc, tries, face_norms, vert_norms, dist = unisample_pnts(sess, ops, -1, batch_data, None, FLAGS.res, nums, threshold=FLAGS.uni_thresh, stdratio=FLAGS.stdratio, stdlwb=FLAGS.stdlwb[0], stdupb=FLAGS.stdupb[0])
            save_norm(loc, norm, os.path.join(outdir, "uni.ply"))
        elif FLAGS.unitype == "ball":
            loc, norm, std, weights, gt_pc, tries, face_norms, vert_norms, dist = ballsample_pnts(sess, ops, FLAGS.anglenums, -1, batch_data, None, nums, threshold=FLAGS.uni_thresh, stdratio=FLAGS.stdratio, stdlwb=FLAGS.stdlwb[0], stdupb=FLAGS.stdupb[0])
            save_norm(loc, norm, os.path.join(outdir, "ball.ply"))
        if FLAGS.restore_surfmodel != "":
            sess = load_model_strict(sess, ops["saver"], FLAGS.restore_surfmodel)

        for i in range(FLAGS.rounds):
            if FLAGS.rounds > 2 and i == (FLAGS.rounds - 2):
                weightform = "even"
                num_ratio = 1
            else:
                num_ratio = FLAGS.num_ratio
            nums = nums * num_ratio
            pc = sample_from_MM(loc, norm, std, weights, nums, num_ratio, dist, distr=distr, gridsize=FLAGS.gridsize)
            batch_data['pnts'] = np.array([pc])
            # print("batch_data['pnts'].shape", batch_data['pnts'].shape)
            loc, norm, std, weights, dist = nearsample_pnts(sess, ops, i, batch_data, tries, face_norms, vert_norms, stdratio=FLAGS.stdratio, weightform=weightform, stdlwb=FLAGS.stdlwb[i+1], stdupb=FLAGS.stdupb[i+1])
            np.savetxt(os.path.join(outdir, "surf_samp{}.txt".format(i)), pc, delimiter=";")
            save_norm(loc, norm, os.path.join(outdir, "surf_{}.ply".format(i)))
        sys.stdout.flush()
        print("finished inference {}/{}".format(batch_idx+1, num_batches))



def gt_test_one_epoch(raw_dirs, TEST_DATASET):
    num_batches = int(len(TEST_DATASET) / FLAGS.batch_size)
    print('num_batches', num_batches)
    log_string(FLAGS.log_dir)
    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = TEST_DATASET.fetch()
        view_id = batch_data["view_id"][0]
        outdir = os.path.join(FLAGS.outdir, batch_data["cat_id"][0], batch_data["obj_nm"][0], str(view_id), FLAGS.unitype)
        os.makedirs(outdir, exist_ok=True)
        nums = FLAGS.initnums
        weightform = FLAGS.weightform
        distr = FLAGS.distr
        modelfile = os.path.join(raw_dirs['norm_mesh_dir'], batch_data["cat_id"][0], batch_data["obj_nm"][0], "pc_norm.obj")

        if FLAGS.unitype == "uni":
            loc, norm, std, weights, gt_pc, tries, face_norms, vert_norms, dist = unisample_pnts(None, None, -1, batch_data, modelfile, FLAGS.res, nums, threshold=FLAGS.uni_thresh, stdratio=FLAGS.stdratio, stdlwb=FLAGS.stdlwb[0], stdupb=FLAGS.stdupb[0])
            save_norm(loc, norm, os.path.join(outdir, "uni.ply"))
        elif FLAGS.unitype == "ball":
            loc, norm, std, weights, gt_pc, tries, face_norms, vert_norms, dist = ballsample_pnts(None, None, -1,  FLAGS.anglenums, batch_data, modelfile, nums, threshold=FLAGS.uni_thresh, stdratio=FLAGS.stdratio, stdlwb=FLAGS.stdlwb[0], stdupb=FLAGS.stdupb[0])
            save_norm(loc, norm, os.path.join(outdir, "ball.ply"))

        for i in range(FLAGS.rounds):
            num_ratio = FLAGS.num_ratio
            nums = nums * num_ratio
            pc = sample_from_MM(loc, norm, std, weights, nums, num_ratio, dist, distr=distr, gridsize=FLAGS.gridsize)
            batch_data['pnts'] = np.array([pc])
            print("batch_data['pnts'].shape", batch_data['pnts'].shape)
            loc, norm, std, weights, dist = nearsample_pnts(None, None, i, batch_data, tries, face_norms, vert_norms, stdratio=FLAGS.stdratio, weightform=weightform, stdlwb=FLAGS.stdlwb[i+1], stdupb=FLAGS.stdupb[i+1])
            # print("pc.shape", pc.shape, "loc.shape", loc.shape, "std.shape", std.shape, "weights.shape", weights.shape)
            np.savetxt(os.path.join(FLAGS.outdir, "surf_samp{}.txt".format(i)), pc, delimiter=";")
            save_norm(loc, norm, os.path.join(outdir, "surf_{}.ply".format(i)))
        sys.stdout.flush()
        print("finished gt {}/{}".format(batch_idx+1, num_batches))
    print("Done!")

def rectify(norm, right_drct):
    cosim = np.dot(norm, right_drct)
    norm = -norm if cosim < 0 else norm
    return norm

def randomneg(norm):
    ind = np.array(sample([i for i in range(norm.shape[0])], norm.shape[0]//2))
    norm[ind] = -norm[ind]
    print(ind.shape, norm.shape)
    return norm

def get_right_drct(norm, ind, nnindices, nndistances, flag):
    marked_nn_ind = nnindices[flag[nnindices] > 0]
    if marked_nn_ind.shape[0] == 1:
        return norm[marked_nn_ind[0]]
    return np.mean(norm[marked_nn_ind], axis = 0)

def linearRect(loc, norm):
    indarr = np.arange(loc.shape[0])
    flag = np.zeros(loc.shape[0], dtype=int)
    dist = dm.get_metric("euclidean")
    dist_matrix = dist.pairwise(loc)
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(loc)
    nndistances, nnindices = nbrs.kneighbors(loc)
    # dist_sorted_ind = np.argsort(dist_matrix, axis=1)[:,1:]
    flag_closest_rected = np.ones_like(flag, dtype=int) * -1
    flag_closest_rected_dist = np.ones_like(flag, dtype=float) * 10
    ind = np.argmax(np.linalg.norm(loc, axis=1))
    right_drct = loc[ind] / np.linalg.norm(loc[ind])
    for i in range(loc.shape[0]):
        if i != 0:
            unmarkind = np.argmin(flag_closest_rected_dist[flag<1])
            ind = indarr[flag<1][unmarkind]
            right_drct = get_right_drct(norm, ind, nnindices[ind], nndistances[ind], flag)
            # print(ind, flag_closest_rected[ind], "flag_closest_rected_dist[ind]", flag_closest_rected_dist[ind])
        norm[ind] = rectify(norm[ind], right_drct)
        flag[ind] = 1
        flag_closest_rected_dist_new = np.minimum(dist_matrix[:, ind],flag_closest_rected_dist)
        flag_closest_rected = np.where(flag_closest_rected_dist_new<flag_closest_rected_dist, ind, flag_closest_rected)
        flag_closest_rected_dist = flag_closest_rected_dist_new
    return norm

def inference_batch(sess, ops, roundnum, batch_data):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    log_string(str(datetime.now()))
    SPLIT_SIZE = int(np.ceil(batch_data['pnts'].shape[1] / NUM_INPUT_POINTS))
    totalnum = batch_data['pnts'].shape[1]
    extra_pts = np.zeros((1, SPLIT_SIZE * NUM_INPUT_POINTS - totalnum, 3), dtype=np.float32)
    batch_points = np.concatenate([batch_data['pnts'], extra_pts], axis = 1).reshape((SPLIT_SIZE, 1, -1, 3))
    pred_ivt_lst = []
    pred_drct_lst = []
    tic = time.time()
    for sp in range(SPLIT_SIZE):
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['pnts']: batch_points[sp,:],
                     ops['input_pls']['imgs']: batch_data['imgs'],
                     ops['input_pls']['obj_rot_mats']: batch_data['obj_rot_mats'],
                     ops['input_pls']['trans_mats']: batch_data['trans_mats']}
        output_list = [ops['end_points']['pred_ivts_xyz'], ops['end_points']['pred_ivts_dist'],
                       ops['end_points']['pred_ivts_direction'], ops['end_points']['imgs']]
        loss_list = []
        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)
        pred_xyz_val, pred_dist_val, pred_direction_val, _ = outputs[:]
        pred_ivt_lst.append(pred_xyz_val)
        pred_drct_lst.append(pred_direction_val)
    pred_ivts = np.concatenate(pred_ivt_lst, axis=1).reshape(-1,3)[:totalnum,:]
    pred_directions = np.concatenate(pred_drct_lst, axis=1).reshape(-1,3)[:totalnum,:]
    outstr = ' -----rounds %d, %d points ------ ' % (roundnum, totalnum)
    outstr += ' time per obj: %.02f, ' % (time.time() - tic)
    log_string(outstr)
    return pred_ivts, pred_directions

def get_unigrid(ivt_res):
    grids = int(1 / ivt_res)
    x_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    y_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    z_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)

def get_normalized_mesh(model_file):
    total = 16384 * 5
    print("trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum += area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all = np.zeros((0, 3), dtype=np.float32)
    all_face_normals = np.zeros((0, 3), dtype=np.float32)
    all_vert_normals = np.zeros((0, 3, 3), dtype=np.float32)
    all_tries = np.zeros((0, 3, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        vert_ind = mesh.faces.reshape(-1)
        # print("end sample surface")
        all_tries = np.concatenate([all_tries, mesh.vertices[vert_ind].reshape([-1, 3, 3])], axis=0)
        all_face_normals = np.concatenate([all_face_normals, mesh.face_normals], axis=0)
        all_vert_normals = np.concatenate([all_vert_normals, mesh.vertex_normals[vert_ind].reshape([-1, 3, 3])], axis=0)
        points_all = np.concatenate([points_all, points], axis=0)
    return points_all, all_tries, all_face_normals, all_vert_normals


def unisample_pnts(sess, ops, roundnum, batch_data, modelfile, res, num, threshold=0.1, stdratio=10, stdlwb=0.0, stdupb=0.1):
    gt_pnts, tries, face_norms, vert_norms = None, None, None, None
    unigrid = get_unigrid(res)
    if not FLAGS.unionly:
        inds = np.random.choice(unigrid.shape[0], num * 8)
        unigrid = unigrid[inds]  # uni_ivts = ct.gpu_calculate_ivt(unigrid, tries, gpu)
    batch_data["pnts"] = np.array([unigrid])
    if sess is None:
        gt_pnts, tries, face_norms, vert_norms = get_normalized_mesh(modelfile)
        uni_ivts, uni_inter_norm = ct.gpu_calculate_ivt_norm(batch_data["pnts"][0], tries, face_norms, vert_norms, int(FLAGS.gpu), tries.shape[0]>700)
        uni_dist = np.linalg.norm(uni_ivts, axis=1, keepdims=True)
        uni_drcts = uni_inter_norm if FLAGS.norminter else -uni_ivts / uni_dist
        uni_dist = uni_dist.reshape((-1))
        print("uni_ivts.shape, uni_drcts.shape", uni_ivts.shape, uni_drcts.shape)
    else:
        uni_ivts, uni_drcts = inference_batch(sess, ops, roundnum, batch_data)
        uni_drcts = -uni_drcts
        uni_dist = np.linalg.norm(uni_ivts, axis=1)
    ind = uni_dist <= threshold
    uni_drcts = uni_drcts[ind]
    dist, uni_place, std = cal_std_loc(unigrid[ind], uni_ivts[ind], stdratio, stdlwb=stdlwb, stdupb=stdupb)
    return uni_place, -uni_drcts, std, np.full(std.shape[0], 1), gt_pnts, tries, face_norms, vert_norms, dist

def get_ballgrid(angles_num):
    phi = np.linspace(0, np.pi, num=angles_num)
    theta = np.linspace(0, 2 * np.pi, num=angles_num)
    x = np.outer(np.sin(theta), np.cos(phi)).reshape(-1)
    y = np.outer(np.sin(theta), np.sin(phi)).reshape(-1)
    z = np.outer(np.cos(theta), np.ones_like(phi)).reshape(-1)
    return np.stack([x, y, z], axis=1)

def ballsample_pnts(sess, ops, roundnum, angles_num, batch_data, modelfile, num, threshold=0.1, stdratio=10, stdlwb=0.0, stdupb=0.1):
    gt_pnts, tries, face_norms, vert_norms = None, None, None, None
    ballgrid = get_ballgrid(angles_num)
    if not FLAGS.unionly:
        inds = np.random.choice(ballgrid.shape[0], num * 8)
        ballgrid = ballgrid[inds]
    batch_data["pnts"] = np.array([ballgrid])
    if sess is None:
        gt_pnts, tries, face_norms, vert_norms = get_normalized_mesh(modelfile)
        ball_ivts, ball_inter_norm = ct.gpu_calculate_ivt_norm(batch_data["pnts"][0], tries, face_norms, vert_norms, int(FLAGS.gpu), tries.shape[0]>700)
        ball_dist = np.linalg.norm(ball_ivts, axis=1, keepdims=True)
        ball_drcts = ball_inter_norm if FLAGS.norminter else -ball_ivts / ball_dist
        ball_dist = ball_dist.reshape((-1))
        print("ball_ivts.shape, ball_drcts.shape", ball_ivts.shape, ball_drcts.shape)
    else:
        ball_ivts, ball_drcts = inference_batch(sess, ops, roundnum, batch_data)
        ball_drcts=-ball_drcts
        ball_dist = np.linalg.norm(uni_ivts, axis=1)
    # ind = ball_dist <= threshold
    # ivts = ball_drcts[ind]
    dist, ball_place, std = cal_std_loc(ballgrid, ball_ivts, stdratio, stdlwb=stdlwb, stdupb=stdupb)
    return ball_place, ball_drcts, std, np.full(std.shape[0], 1), gt_pnts, tries, face_norms, vert_norms, dist


def cal_std_loc(pnts, ivts, stdratio, stdlwb=0.0, stdupb=0.1):
    loc = pnts + ivts
    # print("ivts.shape",ivts.shape)
    dist = np.linalg.norm(ivts, axis=1)
    print("stdupb,stdlwb",stdupb,stdlwb)
    std = np.minimum(stdupb, np.maximum(np.minimum(stdlwb, dist), dist / stdratio))
    # print("np.amax(std)",np.amax(std))
    return dist, loc, std


def sample_from_MM(locs, norm, std, weights, nums, num_ratio, dist, distr="gaussian", gridsize=-1.0):
    if np.sum(std) != 0:
        print(locs.shape, norm.shape, std.shape, weights.shape)
        times = nums*2 // locs.shape[0]
        if times > 0:
            locs = np.tile(locs, (times, 1))
            norm = np.tile(norm, (times, 1))
            std = np.tile(std, (times))
            dist = np.tile(dist, (times))
            weights = np.tile(weights, (times))
        print(locs.shape, norm.shape, std.shape, weights.shape)
        sampled_pnts = np.zeros((locs.shape[0], 3))
        if distr == "gaussian" or distr == "ball":
            std = np.tile(np.expand_dims(std, axis=1), (1, 3))
            for i in range(nums):
                ind = inds[i]
                if distr == "gaussian":
                    p_loc = np.random.normal(locs[ind], std[ind])
                elif distr == "ball":
                    p_loc = ball_sample(locs[ind], std[ind])
                sampled_pnts[i, :] = p_loc
        else:
            back = float(distr)
            sampled_pnts[:,:] = normback_sample(locs, norm, std, np.sqrt(np.maximum(0,2*np.multiply(dist,std)- np.square(std)))) # dist*0.8)
            print("max shift",np.amax(np.sqrt(2*np.multiply(dist,std)- np.square(std))))

        if gridsize > 0:
            print("weights.shape", weights.shape)
            weights = grid_weight(sampled_pnts, gridsize, weights)
            print("weights.shape", weights.shape)
        weights = weights / np.sum(weights)
        print("weights.shape", weights.shape)
        print("locs.shape", locs.shape)
        print("std.shape", std.shape)
        print("nums", nums)
        inds = np.random.choice(weights.shape[0], nums, p=weights)
        return sampled_pnts[inds]
    else:
        print("std == 0, no samples")
        return locs

def grid_weight(loc, gridsize, weights):
    minXYZ, maxXYZ = np.min(loc, axis=0), np.max(loc, axis=0)
    xyzDims = np.floor((maxXYZ - minXYZ) / gridsize)
    norm_loc = loc - minXYZ
    norm_loc_xyzind = np.floor(norm_loc / gridsize)
    norm_loc_ind = norm_loc_xyzind[:,0] * xyzDims[1] * xyzDims[2] + norm_loc_xyzind[:,1] * xyzDims[2] + norm_loc_xyzind[:,2]
    unique, reverse_index, counts = np.unique(norm_loc_ind, return_inverse=True, return_counts=True)
    weight_count = counts[reverse_index]
    weights = weights / (1.0 * weight_count)
    return weights


def ball_sample(xyzmean, radius):
    uvw = np.random.normal(np.array([0, 0, 0]), np.array([1, 1, 1]))
    e = np.random.exponential(0.5)
    denom = (e + np.dot(uvw, uvw)) ** 0.5
    return xyzmean + uvw / denom * radius

def normback_sample(xyzmean, norm, back, radius):
    xyzmean = xyzmean + np.multiply(norm,np.expand_dims(back, axis=1))
    num = xyzmean.shape[0]
    r = np.multiply(radius, np.sqrt(np.random.uniform(0, 1, size=num)))
    a = np.random.uniform(0, np.pi * 2, size=num)
    xyzround = np.expand_dims(np.stack([np.multiply(r, np.cos(a)), np.multiply(r, np.sin(a)), np.zeros_like(r)], axis=1), axis=2)
    R = z_norm_matrix(norm)
    print(R.shape, xyzround.shape)
    xyzround = np.matmul(R, xyzround).reshape((-1,3))
    return xyzmean + xyzround

def z_norm_matrix(norm):
    bs = norm.shape[0]
    z_b = np.repeat(np.array([(0., 0., 1.)]), bs, axis=0)
    normal = np.cross(z_b, norm)
    print("normal of norm rotate:", normal.shape)
    sinal = np.linalg.norm(normal,axis=1,keepdims=True)
    cosal = np.sum(np.multiply(z_b, norm),axis=1,keepdims=True)
    # sinalsqrrev = 1/np.square(np.maximum(1e-5, sinal))
    ux = normal[:,[0]]
    uy = normal[:,[1]]
    uz = normal[:,[2]]
    zr = np.zeros_like(ux)
    W = np.concatenate([zr, -uz, uy, uz, zr, -ux, -uy, ux, zr], axis=1).reshape((-1,3,3))
    I = np.repeat(np.expand_dims(np.identity(3), axis = 0), bs, axis=0)
    Wsqr = np.matmul(W, W)
    C = 1/np.maximum(1e-5,1+cosal)
    R = I + W + np.expand_dims(C, axis=2) * Wsqr
    return R

def nearsample_pnts(sess, ops, roundnum, batch_data, tries, face_norms, vert_norms, stdratio=10, weightform="reverse", stdlwb=0.04, stdupb=0.1):
    if sess is None:
        surface_ivts, surf_inter_norm = ct.gpu_calculate_ivt_norm(batch_data["pnts"][0], tries, face_norms, vert_norms, int(FLAGS.gpu), tries.shape[0]>700)
        dist = np.linalg.norm(surface_ivts, axis=1, keepdims=True)
        surf_norm = surf_inter_norm if FLAGS.norminter else -surface_ivts / dist
        print("surface_ivts.shape, surface_ivts.shape", surface_ivts.shape, surface_ivts.shape)
    else:
        surface_ivts, surf_norm = inference_batch(sess, ops, roundnum, batch_data)
        surf_norm=-surf_norm
    # print("batch_data[pnts][0], surface_ivts",batch_data["pnts"][0].shape, surface_ivts.shape)
    dist, surface_place, std = cal_std_loc(batch_data["pnts"][0], surface_ivts, stdratio, stdlwb=stdlwb, stdupb=stdupb)
    if weightform == "even":
        weights = np.full(std.shape[0], 1)
    elif weightform == "reverse":
        weights = 1.0 / np.maximum(dist, 5e-3)

    return surface_place, surf_norm, std, weights, dist



if __name__ == "__main__":
    # nohup python -u inference.py --gpu 3 --img_feat_onestream --category chair  --restore_model ../train/checkpoint/onestream_small_spheregrid/chair_vgg_16_010000 --restore_surfmodel ../train/checkpoint/onestream_small_surf/chair_vgg_16_010000 --outdir  chair_drct_even_surfonly_uni --unionly --unitype uni --XYZ &> global_direct_chair_surf_evenweight_uni.log &


    # nohup python -u inference.py --gt --outdir  gt_noerrBall --unionly --stdupb 0.3 0.3 0.2 0.1 0.05 --stdlwb 0.01 0.01 0.01 0.01 0.01 &> gt_uni.log &

    #   nohup python -u inference.py --gt --outdir  gt_shirt1 --unionly --norminter --stdupb 0.3 0.3 0.2 0.1 0.05 --stdlwb 0.01 0.0 0.0 0.0 0.0 &> shirt.log &

    # full set

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
    parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
    parser.add_argument('--num_pnts', type=int, default=0, help='Point Number [default: 2048]')
    parser.add_argument('--uni_num', type=int, default=0, help='Point Number [default: 2048]')
    parser.add_argument('--sphere_num', type=int, default=0, help='Point Number [default: 2048]')
    parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
    parser.add_argument("--beta1", type=float, dest="beta1", default=0.5, help="beta1 of adams")
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
    parser.add_argument('--img_h', type=int, default=137, help='Image Height')
    parser.add_argument('--img_w', type=int, default=137, help='Image Width')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
    parser.add_argument('--restore_model', default='', help='restore_model')  # checkpoint/sdf_2d3d_sdfbasic2_nowd
    parser.add_argument('--restore_surfmodel', default='',
                        help='restore_model surface only')  # ../models/CNN/pretrained_model/vgg_16.ckpt
    parser.add_argument('--train_lst_dir', default=lst_dir, help='train mesh data list')
    parser.add_argument('--valid_lst_dir', default=lst_dir, help='test mesh data list')
    parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--weight_type', type=str, default="ntanh")
    parser.add_argument('--dir_loss_weight', type=float, default=0.1)
    parser.add_argument('--img_feat_onestream', action='store_true')
    parser.add_argument('--img_feat_twostream', action='store_true')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--alpha', action='store_true')
    parser.add_argument('--augcolorfore', action='store_true')
    parser.add_argument('--augcolorback', action='store_true')
    parser.add_argument('--backcolorwhite', action='store_true')
    parser.add_argument('--rot', action='store_true')
    parser.add_argument('--XYZ', action='store_true')
    parser.add_argument('--cam_est', action='store_true')
    parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
    parser.add_argument('--multi_view', action='store_true')
    parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
    parser.add_argument('--rounds', type=int, default=4, help='how many rounds of ivf')
    parser.add_argument('--uni_thresh', type=float, default=0.1, help='threshold for uniform sampling')
    parser.add_argument('--res', type=float, default=0.01, help='cube resolution')
    parser.add_argument('--anglenums', type=int, default=200, help='angle resolution')
    parser.add_argument('--max_epoch', type=int, default=1, help='angle resolution')
    parser.add_argument('--initnums', type=int, default=8096, help='initial sampled uni point numbers')
    parser.add_argument('--num_ratio', type=int, default=2, help='point numbers expansion each round')
    parser.add_argument('--stdratio', type=int, default=4, help='')
    parser.add_argument('--stdupb', nargs='+', action='store', default=[0.1, 0.1, 0.03, 0.01, 0])
    parser.add_argument('--stdlwb', nargs='+', action='store', default=[0.08, 0.04, 0.003, 0.001, 0])
    # parser.add_argument('--stdupb', nargs='+', action='append', default=[0, 0])
    # parser.add_argument('--stdlwb', nargs='+', action='append', default=[0, 0])
    parser.add_argument('--distr', type=str, default="0.1", help='local points sampling: gaussian or ball')
    parser.add_argument('--weightform', type=str, default="even", help='GMM weight: even or reverse')
    parser.add_argument('--gridsize', type=float, default=0.02, help='GMM weight: even or reverse')
    parser.add_argument('--outdir', type=str, default="./", help='out_dir')
    parser.add_argument('--unitype', type=str, default="ball", help='out_dir')
    parser.add_argument('--unionly', action='store_true')
    parser.add_argument('--gt', action='store_true')
    parser.add_argument('--norminter', action='store_true')
    parser.add_argument('--smaller', action='store_true')
    parser.add_argument('--act', type=str, default="relu")
    parser.add_argument('--encoder', type=str, default='vgg_16',
                        help='encoder model: vgg_16, resnet_v1_50, resnet_v1_101, resnet_v2_50, resnet_v2_101')
    parser.add_argument('--wd', type=float, default=1e-6, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--decoderskip', action='store_true')
    parser.add_argument('--lossw', nargs='+', action='store', default=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                        help="xyz, locnorm, locsqrnorm, dist, dirct, drct_abs")
    parser.add_argument('--distlimit', nargs='+', action='store', type=str,
                        default=[1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2,
                                 0.18, 0.18, 0.16, 0.16, 0.14, 0.14, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.06, 0.06, 0.05,
                                 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, -0.01])
    parser.add_argument('--surfrange', nargs='+', action='store', default=[0.0, 0.15], help="lower bound, upperbound")
    FLAGS = parser.parse_args()
    FLAGS.stdupb = [float(i) for i in FLAGS.stdupb]
    FLAGS.stdlwb = [float(i) for i in FLAGS.stdlwb]
    FLAGS.lossw = [float(i) for i in FLAGS.lossw]
    FLAGS.distlimit = [float(i) for i in FLAGS.distlimit]
    FLAGS.surfrange = [float(i) for i in FLAGS.surfrange]
    print(FLAGS)



    # testarray = normback_sample(np.zeros((100,3)), np.tile(np.array([[0, 0.5, np.sqrt(3)*0.5]]), (100,1)), np.ones(100), np.ones(100))
    #
    # print(testarray)
    # np.savetxt("./test.txt", testarray, delimiter=";")
    # a = np.array([[[1,1,1],[2,2,2],[3,3,3]],[[5,5,5],[6,6,6],[7,7,7]],[[9,9,9],[11,11,11],[13,13,13]], [[93,90,90],[110,110,110],[130,130,130]]])
    # b = np.array([1,3])
    # print(np.take(a, b, axis=0))
    #
    # exit()


    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

    if not os.path.exists(FLAGS.log_dir): os.makedirs(FLAGS.log_dir)

    RESULT_PATH = os.path.join(FLAGS.log_dir, 'test_results')
    if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

    os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], FLAGS.log_dir))
    os.system('cp inference.py %s' % (FLAGS.log_dir))
    LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_test.txt'), 'w')
    LOG_FOUT.write(str(FLAGS) + '\n')

    MAXOUT = FLAGS.initnums * FLAGS.num_ratio ** (FLAGS.rounds - 1)
    NUM_INPUT_POINTS = min(MAXOUT, 274625)
    print("NUM_INPUT_POINTS", NUM_INPUT_POINTS)

    BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)

    if FLAGS.category != "all":
        cats = {
            FLAGS.category:cats[FLAGS.category]
        }

    TEST_LISTINFO = []
    cats_limit = {}

    cat_ids = []
    if FLAGS.category == "all":
        for key, value in cats.items():
            cat_ids.append(value)
            cats_limit[value] = 0
    else:
        cat_ids.append(cats[FLAGS.category])
        cats_limit[cats[FLAGS.category]] = 0

    for cat_id in cat_ids:
        test_lst = os.path.join(FLAGS.test_lst_dir, cat_id + "_test.lst")
        with open(test_lst, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                for render in range(24):
                    cats_limit[cat_id] += 1
                    TEST_LISTINFO += [(cat_id, line.strip(), render)]

    # TEST_LISTINFO += [("03001627", "17e916fc863540ee3def89b32cef8e45", 11)]
    # TEST_LISTINFO += [("03001627", "1be38f2624022098f71e06115e9c3b3e", 0)]
    # TEST_LISTINFO += [("02691156", "1beb0776148870d4c511571426f8b16d", 0)]

    info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'ivt_dir': raw_dirs["ivt_mani_dir"]}
    if FLAGS.cam_est:
        info['rendered_dir'] = raw_dirs["renderedh5_dir_est"]

    print(info)

    TEST_DATASET = data_ivt_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit, shuffle=False)


    # batch_data={"cat_id":["02691156"],"obj_nm":["1beb0776148870d4c511571426f8b16d"]}
    # model_file = os.path.join("/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/", batch_data["cat_id"][0], batch_data["obj_nm"][0], "isosurf.obj")
    #
    # command_str = "cp " + model_file + " ./" + FLAGS.outdir + "/"
    # print("command:", command_str)
    # os.system(command_str)
    # batch_data = {"cat_id": ["000"], "obj_nm": ["shirt3"]}
    # model_file = "./shirt.obj"
    # ct.get_normalize_mesh(model_file, "./", "./", "./", 0)
    # model_file = "./pc_norm.obj"
    # batch_data["model_file"] = model_file
    os.makedirs(FLAGS.outdir, exist_ok=True)

    if FLAGS.gt:
        gt_test_one_epoch(raw_dirs, TEST_DATASET)
    else:
        test(raw_dirs, TEST_DATASET)
    print("done!")
    LOG_FOUT.close()
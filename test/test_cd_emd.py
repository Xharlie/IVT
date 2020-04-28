import argparse
import numpy as np
import random
import tensorflow as tf
import socket
import pymesh
import os
import sys
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'data'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import h5py
import models.tf_ops.approxmatch.tf_approxmatch as tf_approxmatch
import models.tf_ops.nn_distance.tf_nndistance as tf_nndistance
import create_file_lst
slim = tf.contrib.slim
import trimesh
import time

parser = argparse.ArgumentParser()
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

parser.add_argument('--store', action='store_true')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg global embedding dimensions')
parser.add_argument('--num_points', type=int, default=1, help='Point Number [default: 2048]')
parser.add_argument('--alpha', action='store_true')
parser.add_argument('--rot', action='store_true')
parser.add_argument('--tanh', action='store_true')
parser.add_argument('--cat_limit', type=int, default=168000, help="balance each category, 1500 * 24 = 36000")
parser.add_argument('--sdf_res', type=int, default=64, help='sdf grid')
parser.add_argument('--binary', action='store_true')

parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
parser.add_argument('--set', type=str, default='test')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 32]')
parser.add_argument('--log_dir', default='checkpoint/exp_200', help='Log dir [default: log]')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--num_sample_points', type=int, default=2048, help='Sample Point Number for each obj to test[default: 2048]')
parser.add_argument('--threedcnn', action='store_true')
parser.add_argument('--img_feat_onestream', action='store_true')
parser.add_argument('--img_feat_twostream', action='store_true')
parser.add_argument('--category', default="all", help='Which single class to train on [default: None]')
parser.add_argument('--view_num', type=int, default=24, help="how many views do you want to create for each obj")
parser.add_argument('--cam_est', action='store_true')
parser.add_argument('--revert', action='store_true')
parser.add_argument('--cal_dir', type=str, default="", help="target obj directory that needs to be tested")
parser.add_argument('--unitype', type=str, default="ball", help="target obj directory that needs to be tested")
parser.add_argument('--round', type=int, default=0, help="target obj directory that needs to be tested")
parser.add_argument('--fthresholds', nargs='+', action='store', default=[0.01, 0.02, 0.05], help="lower bound, upperbound")
parser.add_argument('--ioudims', nargs='+', action='store', default=[64,110], help="lower bound, upperbound")

FLAGS = parser.parse_args()
print('pid: %s'%(str(os.getpid())))
print(FLAGS)
FLAGS.fthresholds = [float(i) for i in FLAGS.fthresholds]
FLAGS.ioudims = [int(i) for i in FLAGS.ioudims]

EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
GPU_INDEX = FLAGS.gpu
PRETRAINED_MODEL_PATH = FLAGS.log_dir
LOG_DIR = FLAGS.log_dir
SDF_WEIGHT = 10.

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

RESULT_PATH = os.path.join(LOG_DIR, 'test_results_allpts')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

IMG_SIZE = FLAGS.img_h
VV =False
HOSTNAME = socket.gethostname()

TEST_LISTINFO = []


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)



def load_model(sess, LOAD_MODEL_FILE, prefixs, strict=False):

    vars_in_pretrained_model = dict(checkpoint_utils.list_variables(LOAD_MODEL_FILE))
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
    saver = tf.train.Saver(vars_in_defined_model)
    saver.restore(sess, LOAD_MODEL_FILE)
    print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))

    return True

def build_file_dict(pred_dir_cat):
    file_dict = {}
    for obj_id in os.listdir(pred_dir_cat):
        obj_dir_path = os.path.join(pred_dir_cat, obj_id) #
        for view_id in os.listdir(obj_dir_path):
            obj_view_dir_path = os.path.join(obj_dir_path, view_id, FLAGS.unitype)
            full_path = os.path.join(obj_view_dir_path, "surf_{}.h5".format(FLAGS.round) if FLAGS.round >=0 else "{}.h5".format(FLAGS.unitype))
            if obj_id in file_dict.keys():
                file_dict[obj_id].append(full_path)
            else:
                file_dict[obj_id] = [full_path]
    return file_dict

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()

def cd_emd_all(cats, pred_dir, gt_dir, test_lst_dir, param_dir):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_{}.lst".format(FLAGS.set))
        cd_emd_cat(cat_id, cat_nm, pred_dir_cat, gt_dir_cat, test_lst_f, param_dir)
    print("done!")

def save_all_cat_gt_pnt(cats, gt_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        gt_dir_cat = os.path.join(gt_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        sample_save_gt_pnt(cat_id, cat_nm, gt_dir_cat, test_lst_f)
    print("done!")

def sample_save_gt_pnt(cat_id, cat_nm, gt_dir_cat, test_lst_f):
    count = 0
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        count += 1
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            obj_path = os.path.join(gt_dir_cat, obj_id, "isosurf.obj")
            # pred_path_lst = pred_dict[obj_id]
            verts_batch = np.zeros((FLAGS.num_sample_points, 3), dtype=np.float32)
            mesh1 = pymesh.load_mesh(obj_path)
            if mesh1.vertices.shape[0] > 0:
                choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                verts_batch = mesh1.vertices[choice, ...]
            savefn = os.path.join(gt_dir_cat, obj_id, "pnt_{}.txt".format(FLAGS.num_sample_points))
            np.savetxt(savefn, verts_batch, delimiter=',')
            print("saved gt pnt of {} at {}".format(obj_id, savefn))


def save_all_cat_pred_pnt(cats, pred_dir, test_lst_dir):
    for cat_nm, cat_id in cats.items():
        pred_dir_cat = os.path.join(pred_dir, cat_id)
        test_lst_f = os.path.join(test_lst_dir, cat_id+"_test.lst")
        sample_save_pred_pnt(cat_id, cat_nm, pred_dir_cat, test_lst_f)
    print("done!")

def sample_save_pred_pnt(cat_id, cat_nm, pred_dir, test_lst_f):
    pred_dict = build_file_dict(pred_dir)
    with open(test_lst_f, "r") as f:
        test_objs = f.readlines()
        for obj_id in test_objs:
            obj_id = obj_id.rstrip('\r\n')
            pred_path_lst = pred_dict[obj_id]
            verts_batch = np.zeros((FLAGS.view_num, FLAGS.num_sample_points, 3), dtype=np.float32)
            for i in range(len(pred_path_lst)):
                pred_mesh_fl = pred_path_lst[i]
                mesh1 = pymesh.load_mesh(pred_mesh_fl)
                if mesh1.vertices.shape[0] > 0:
                    choice = np.random.randint(mesh1.vertices.shape[0], size=FLAGS.num_sample_points)
                    verts_batch[i, ...] = mesh1.vertices[choice, ...]
                savedir = os.path.join(os.path.dirname(pred_dir),"pnt_{}_{}".format(FLAGS.num_sample_points, cat_id))
                os.makedirs(savedir,exist_ok=True)
                view_id = pred_mesh_fl[-6:-4]
                savefn = os.path.join(savedir, "pnt_{}_{}.txt".format(obj_id, view_id))
                print(savefn)
                np.savetxt(savefn, verts_batch[i, ...], delimiter=',')
                print("saved gt pnt of {} at {}".format(obj_id, savefn))



def cd_emd_cat(cat_id, cat_nm, pred_dir_cat, gt_dir, test_lst_f, param_dir):
    pred_dict = build_file_dict(pred_dir_cat)
    sum_cf_loss = 0.
    sum_fcf_loss = 0.
    sum_bcf_loss = 0.
    sum_em_loss = 0.
    sum_f_score = np.zeros((len(FLAGS.fthresholds)))
    sum_ious = np.zeros((len(FLAGS.ioudims)))
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            sampled_pc = tf.placeholder(tf.float32, shape=(FLAGS.batch_size+1, FLAGS.num_sample_points, 3))
            avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em, avg_fcf_loss, avg_bcf_loss, avg_f_scores = get_points_loss(sampled_pc)
            count = 0
            with open(test_lst_f, "r") as f:
                test_objs = f.readlines()
                count+=1
                for obj_id in test_objs:
                    obj_id = obj_id.rstrip('\r\n')
                    src_path = os.path.join(gt_dir, obj_id, "pc_norm.obj")
                    pred_path_lst = pred_dict[obj_id]
                    verts_batch = np.zeros((FLAGS.view_num+1, FLAGS.num_sample_points, 3), dtype=np.float32)
                    verts_batch[0, ...] = sample_mesh2pnts(src_path)
                    pred_path_lst = random.sample(pred_path_lst, FLAGS.view_num)
                    surfpnts_lst=[]
                    for i in range(len(pred_path_lst)):
                        pred_mesh_fl = pred_path_lst[i]
                        surfpnts = get_surfpnts(pred_mesh_fl)
                        surfpnts_lst.append(surfpnts)
                        if surfpnts.shape[0] > 0:
                            choice = np.random.randint(surfpnts.shape[0], size=FLAGS.num_sample_points)
                            verts_batch[i+1, ...] = surfpnts[choice, ...]
                    assert FLAGS.batch_size == FLAGS.view_num, "FLAGS.batch_size {} != FLAGS.view_num {}".format(FLAGS.batch_size, FLAGS.view_num)
                    avg_ious = iou_pymesh_pnt(src_path, surfpnts_lst)
                    if FLAGS.revert:
                        param_path = os.path.join(param_dir, cat_id, obj_id, "pc_norm.txt")
                        times = np.loadtxt(param_path)[-1]
                        verts_batch = verts_batch * times * 0.57
                    feed_dict = {sampled_pc: verts_batch}
                    avg_cf_loss_val, min_cf_loss_val, arg_min_cf_val, avg_em_loss_val, min_em_loss_val, arg_min_em_val, avg_fcf_loss_val,avg_bcf_loss_val, avg_f_scores_val = sess.run([avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em, avg_fcf_loss, avg_bcf_loss, avg_f_scores], feed_dict=feed_dict)
                    # else:
                    #     sum_avg_cf_loss_val = 0.
                    #     min_cf_loss_val = 9999.
                    #     arg_min_cf_val = 0
                    #     sum_avg_em_loss_val = 0.
                    #     min_em_loss_val = 9999.
                    #     arg_min_em_val = 0
                    #     sum_avg_fcf_loss_val = 0
                    #     sum_avg_bcf_loss_val = 0
                    #     for b in range(FLAGS.view_num//FLAGS.batch_size):
                    #         verts_batch_b = np.stack([verts_batch[0, ...], verts_batch[b, ...]])
                    #         feed_dict = {sampled_pc: verts_batch_b}
                    #         avg_cf_loss_val, _, _, avg_em_loss_val, _, _, avg_fcf_loss_val, avg_bcf_loss_val = sess.run([avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em, avg_fcf_loss, avg_bcf_loss], feed_dict=feed_dict)
                    #         sum_avg_cf_loss_val +=avg_cf_loss_val
                    #         sum_avg_em_loss_val +=avg_em_loss_val
                    #         sum_avg_fcf_loss_val +=avg_fcf_loss_val
                    #         sum_avg_bcf_loss_val +=avg_bcf_loss_val
                    #         if min_cf_loss_val > avg_cf_loss_val:
                    #             min_cf_loss_val = avg_cf_loss_val
                    #             arg_min_cf_val = b
                    #         if min_em_loss_val > avg_em_loss_val:
                    #             min_em_loss_val = avg_em_loss_val
                    #             arg_min_em_val = b
                    #     avg_cf_loss_val = sum_avg_cf_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                    #     avg_em_loss_val = sum_avg_em_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                    #     avg_fcf_loss_val = sum_avg_fcf_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                    #     avg_bcf_loss_val = sum_avg_bcf_loss_val / (FLAGS.view_num//FLAGS.batch_size)
                    sum_cf_loss += avg_cf_loss_val
                    sum_em_loss += avg_em_loss_val
                    sum_fcf_loss += avg_fcf_loss_val
                    sum_bcf_loss += avg_bcf_loss_val
                    sum_f_score += avg_f_scores_val
                    sum_ious += avg_ious
                    print(str(count) +  " ",src_path, "avg cf:{}, min_cf:{}, arg_cf view:{}, avg emd:{}, min_emd:{}, arg_em view:{}, avg_f_scores:{}, avg_ious:{} ".
                          format(str(avg_cf_loss_val), str(min_cf_loss_val), str(arg_min_cf_val),
                                 str(avg_em_loss_val), str(min_em_loss_val), str(arg_min_em_val), avg_f_scores_val,avg_ious))
            print("cat_nm:{}, cat_id:{}, avg_cf:{},  avg_fcf:{},  avg_dcf:{}, avg_emd:{}, avg_f_scores:{}, avg_ious:{}".
                  format(cat_nm, cat_id, sum_cf_loss/len(test_objs),sum_fcf_loss/len(test_objs), sum_bcf_loss/len(test_objs), sum_em_loss/len(test_objs), sum_f_score/len(test_objs), sum_ious/len(test_objs)))

def sample_mesh2pnts(model_file):
    total = max(FLAGS.num_sample_points * 5, 16384)
    # print("trimesh_load:", model_file)
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
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        points_all = np.concatenate([points_all, points], axis=0)

    choice = np.random.randint(points_all.shape[0], size=FLAGS.num_sample_points)
    return points_all[choice, ...]


def get_surfpnts(file):
    if file[-1]=="5":
        with h5py.File(file, 'r') as f1:
            surf_pnts =f1["locs"][:]
    return surf_pnts

def get_points_loss(sampled_pc):
    src_pc = tf.tile(tf.expand_dims(sampled_pc[0,:,:], axis=0), (FLAGS.batch_size, 1, 1))
    if sampled_pc.get_shape().as_list()[0] == 2:
        pred = tf.expand_dims(sampled_pc[1,:,:], axis=0)
    else:
        pred = sampled_pc[1:, :, :]
    print(src_pc.get_shape())
    print(pred.get_shape())

    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pred, src_pc)

    cf_forward_views = tf.reduce_mean(dists_forward, axis=1) * 1000
    cf_backward_views = tf.reduce_mean(dists_backward, axis=1) * 1000
    cf_loss_views = cf_forward_views + cf_backward_views
    print("cf_loss_views.get_shape()", cf_loss_views.get_shape())
    avg_cf_loss = tf.reduce_mean(cf_loss_views)
    avg_fcf_loss = tf.reduce_mean(cf_forward_views)
    avg_bcf_loss = tf.reduce_mean(cf_backward_views)
    min_cf_loss = tf.reduce_min(cf_loss_views)
    arg_min_cf = tf.argmin(cf_loss_views, axis=0)

    match = tf_approxmatch.approx_match(src_pc, pred)
    match_cost = tf_approxmatch.match_cost(src_pc, pred, match) * 0.01
    print("match_cost.get_shape()", match_cost.get_shape())

    avg_em_loss = tf.reduce_mean(match_cost)
    min_em_loss = tf.reduce_min(match_cost)
    arg_min_em = tf.argmin(match_cost)


    dists_forward_sqrt = tf.sqrt(tf.maximum(dists_forward, 1e-8))
    dists_backward_sqrt = tf.sqrt(tf.maximum(dists_backward, 1e-8))
    f_scores=[]
    length = 1 if FLAGS.revert else tf.reduce_max(tf.reduce_max(src_pc[0],axis=0) - tf.reduce_min(src_pc[0],axis=0))
    for i in range(len(FLAGS.fthresholds)):
        threshold = FLAGS.fthresholds[i] * length
        precision = tf.reduce_mean(tf.cast(tf.less_equal(dists_forward_sqrt, threshold),dtype=tf.float32), axis=1)
        recall = tf.reduce_mean(tf.cast(tf.less_equal(dists_backward_sqrt, threshold),dtype=tf.float32), axis=1)
        f_score = 2.0*tf.multiply(precision,recall)/tf.maximum(precision+recall, 1e-8)
        # print("f_score.shape", f_score.get_shape().as_list())
        f_scores.append(tf.reduce_mean(f_score))
    f_scores = tf.convert_to_tensor(f_scores)
    return avg_cf_loss, min_cf_loss, arg_min_cf, avg_em_loss, min_em_loss, arg_min_em, avg_fcf_loss,avg_bcf_loss,f_scores

def iou_pymesh_pnt(mesh_src, surfpnts_lst):
    try:
        mesh1 = pymesh.load_mesh(mesh_src)
        iou_lst = []
        for i in range(len(FLAGS.ioudims)):
            # tic = time.time()
            dim = FLAGS.ioudims[i]
            grid1 = pymesh.VoxelGrid(2./dim)
            grid1.insert_mesh(mesh1)
            grid1.create_grid()
            # print("grid:",time.time()-tic)
            # top = np.max(grid1.mesh.vertices, axis=0)
            # bottom = np.min(grid1.mesh.vertices, axis=0)
            # diff = top - bottom
            # length = np.max(diff)
            # print("top {}, bottom {}, diff {}".format(top, bottom, diff))

            ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
            v1 = np.zeros([dim, dim, dim])
            v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1
            iousum = 0
            for j in range(len(surfpnts_lst)):
                pred_pnts = surfpnts_lst[j]
                ind2 = ((pred_pnts + 1.1) / 2.4 * dim).astype(np.int)
                v2 = np.zeros([dim, dim, dim])
                v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1
                intersection = np.sum(np.logical_and(v1, v2))
                union = np.sum(np.logical_or(v1, v2))
                iousum += float(intersection) / union
            iou_lst.append(iousum/len(surfpnts_lst))
            # print("fill:",time.time() - tic)
        return np.array(iou_lst)
    except:
        print("error mesh {} / {}".format(mesh_src, mesh_pred))


def iou_pymesh(mesh_src, mesh_pred, dim=FLAGS.ioudims):
    try:
        mesh1 = pymesh.load_mesh(mesh_src)
        grid1 = pymesh.VoxelGrid(2./dim)
        grid1.insert_mesh(mesh1)
        grid1.create_grid()

        ind1 = ((grid1.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
        v1 = np.zeros([dim, dim, dim])
        v1[ind1[:,0], ind1[:,1], ind1[:,2]] = 1


        mesh2 = pymesh.load_mesh(mesh_pred)
        grid2 = pymesh.VoxelGrid(2./dim)
        grid2.insert_mesh(mesh2)
        grid2.create_grid()

        ind2 = ((grid2.mesh.vertices + 1.1) / 2.4 * dim).astype(np.int)
        v2 = np.zeros([dim, dim, dim])
        v2[ind2[:,0], ind2[:,1], ind2[:,2]] = 1

        intersection = np.sum(np.logical_and(v1, v2))
        union = np.sum(np.logical_or(v1, v2))
        return [float(intersection) / union, mesh_pred]
    except:
        print("error mesh {} / {}".format(mesh_src, mesh_pred))


if __name__ == "__main__":
    cats_all = {
        "watercraft": "04530566",
        "rifle": "04090263",
        "display": "03211117",
        "lamp": "03636649",
        "speaker": "03691459",
        "chair": "03001627",
        "bench": "02828884",
        "cabinet": "02933112",
        "car": "02958343",
        "airplane": "02691156",
        "sofa": "04256520",
        "table": "04379243",
        "phone": "04401088"
    }
    if FLAGS.category == "all":
        cats=cats_all
    elif FLAGS.category == "clean":
        cats ={ "cabinet": "02933112",
                "display": "03211117",
                "speaker": "03691459",
                "rifle": "04090263",
                "watercraft": "04530566"
        }
    else:
        cats={FLAGS.category: cats_all[FLAGS.category]}

    cd_emd_all(cats, FLAGS.cal_dir, raw_dirs["real_norm_mesh_dir"], FLAGS.test_lst_dir, raw_dirs["real_norm_mesh_dir"])

    # 1. test cd_emd for all categories / some of the categories:

    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info(version=1)s
    # cd_emd_all(cats,
    #            "checkpoint/all_best/sdf_2d_sdfproj_twostream_wd_2048_weight10_ftprev_inweight/test_objs/65_0.0",
    #            "/ssd1/datasets/ShapeNet/march_cube_objs_v1/", "/ssd1/datasets/ShapeNet/filelists/",
    #            num_points=FLAGS.num_points, maxnverts=1000000, maxntris=1000000, num_view=4)

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 1 --view_num 8 --num_sample_points 10000 &> ballr1_10000.log &


# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 4 --view_num 8 &> ballr4_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 0 --view_num 8 &> ballr0_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round -1 --view_num 8 &> ballr-1_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 1 --view_num 8 &> ballr1_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 2 --view_num 8 &> ballr2_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 3 --view_num 8 &> ballr3_2048.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new --unitype uni --round 1 --view_num 8 --revert &> ballr1_2048_revert.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2466 --category chair --cal_dir ../inference/inf_new --unitype uni --round 1 --view_num 8 --revert &> ballr1_2466_revert.log &

# nohup python -u test_cd_emd.py --gpu 3 --batch_size 8 --img_feat_twostream  --num_points 2466 --category chair --cal_dir ../inference/inf_new --unitype uni --round 2 --view_num 8 --revert &> ballr2_2466.log &




# nohup python -u test_cd_emd.py --gpu 3 --batch_size 4 --img_feat_twostream  --num_points 2048 --category chair --cal_dir ../inference/inf_new_train --unitype uni --round -1 --view_num 4 --set train &> train_ballr-1_2048.log &
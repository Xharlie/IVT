import argparse
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import os
import cv2
import sys
import time
from tensorflow.contrib.framework.python.framework import checkpoint_utils
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("PID:", os.getpid())
print(os.path.join(BASE_DIR, 'models'))
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'data_load'))
sys.path.append(os.path.join(BASE_DIR, 'preprocessing'))
import model_normalization as model
import data_ivt_h5_queue # as data
import output_utils
import create_file_lst
lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
parser.add_argument('--num_pnts', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--uni_num', type=int, default=512, help='Point Number [default: 2048]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument("--beta1", type=float, dest="beta1",
                    default=0.5, help="beta1 of adams")
# parser.add_argument('--sdf_points_num', type=int, default=32, help='Sample Point Number [default: 2048]')
parser.add_argument('--rounds', type=int, default=5, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd
parser.add_argument('--restore_modelcnn', default='', help='restore_model')#../models/CNN/pretrained_model/vgg_16.ckpt

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

FLAGS = parser.parse_args()
print(FLAGS)


os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

if not os.path.exists(FLAGS.log_dir): os.makedirs(FLAGS.log_dir)

RESULT_PATH = os.path.join(FLAGS.log_dir, 'train_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)

VALID_RESULT_PATH = os.path.join(FLAGS.log_dir, 'valid_results')
if not os.path.exists(VALID_RESULT_PATH): os.mkdir(VALID_RESULT_PATH)

os.system('cp %s.py %s' % (os.path.splitext(model.__file__)[0], FLAGS.log_dir))
os.system('cp train_ivt.py %s' % (FLAGS.log_dir))
LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
BN_DECAY_CLIP = 0.99

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
    train_lst = os.path.join(FLAGS.train_lst_dir, cat_id+"_test.lst")
    with open(train_lst, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            for render in range(24):
                cats_limit[cat_id]+=1
                TEST_LISTINFO += [(cat_id, line.strip(), render)]

info = {'rendered_dir': raw_dirs["renderedh5_dir"],
            'ivt_dir': raw_dirs["ivt_dir"]}
if FLAGS.cam_est:
    info['rendered_dir']= raw_dirs["renderedh5_dir_est"]

print(info)

TEST_DATASET = data_ivt_h5_queue.Pt_sdf_img(FLAGS, listinfo=TEST_LISTINFO, info=info, cats_limit=cats_limit)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*FLAGS.batch_size,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

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
        print( "Model loaded in file: %s" % (LOAD_MODEL_FILE))
    except:
        if strict:
            print( "Fail to load modelfile: %s" % LOAD_MODEL_FILE)
            return False
        else:
            print( "Fail loaded in file: %s" % (LOAD_MODEL_FILE))
            return True

    return True

def test():
    log_string(FLAGS.log_dir)
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            input_pls = model.placeholder_inputs(scope='inputs_pl', FLAGS=FLAGS)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0, name='batch')
            bn_decay = get_bn_decay(batch)
            # tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, is_training_pl, bn=False, FLAGS=FLAGS)
            loss, end_points = model.get_loss(end_points, FLAGS=FLAGS)
            # tf.summary.scalar('loss', loss)

            # Create a session
            config = tf.compat.v1.ConfigProto()
            gpu_options = tf.compat.v1.GPUOptions()#(per_process_gpu_memory_fraction=0.99)
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.compat.v1.Session(config=config)

            merged = None
            test_writer = None

            ##### all
            update_variables = [x for x in tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]


            # Init variables
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
                # Overall
            saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) if ('lr' not in v.name) and ('batch' not in v.name)])
            ckptstate = tf.train.get_checkpoint_state(FLAGS.restore_model)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(FLAGS.restore_model, os.path.basename(ckptstate.model_checkpoint_path))
                try:
                    load_model(sess, LOAD_MODEL_FILE, ['sdfprediction/fold1', 'sdfprediction/fold2', 'vgg_16'],
                               strict=True)
                    # load_model(sess, LOAD_MODEL_FILE, ['sdfprediction','vgg_16'], strict=True)
                    with NoStdStreams():
                        saver.restore(sess, LOAD_MODEL_FILE)
                    print("Model loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    print("Fail to load overall modelfile: %s" % FLAGS.restore_model)
                    exit()

            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'merged': merged,
                   'step': batch,
                   'end_points': end_points}
            best_xyz_diff, best_dist_diff, best_dir_diff = 10000, 10000, 10000
            for epoch in range(FLAGS.max_epoch):
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                xyz_avg_diff, dist_avg_diff, direction_avg_diff = inference_one_epoch(sess, ops, test_writer, saver)


def inference_one_epoch(sess, ops, test_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    num_batches = int(len(TRAIN_DATASET) / FLAGS.batch_size)

    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    tic = time.time()
    fetch_time = 0
    xyz_avg_diff_epoch = 0
    dist_avg_diff_epoch = 0
    direction_avg_diff_epoch = 0
    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = TRAIN_DATASET.fetch()
        fetch_time += (time.time() - start_fetch_tic)
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['pnts']: batch_data['pnts'],
                     ops['input_pls']['ivts']: batch_data['ivts'],
                     ops['input_pls']['imgs']: batch_data['imgs'],
                     ops['input_pls']['obj_rot_mats']: batch_data['obj_rot_mats'],
                     ops['input_pls']['trans_mats']: batch_data['trans_mats']}
        output_list = [ops['train_op'], ops['step'], ops['lr'],  ops['end_points']['pnts_rot'], ops['end_points']['gt_ivts_xyz'], ops['end_points']['gt_ivts_dist'], ops['end_points']['gt_ivts_direction'], ops['end_points']['pred_ivts_xyz'], ops['end_points']['pred_ivts_dist'],ops['end_points']['pred_ivts_direction'], ops['end_points']['sample_img_points'], ops['end_points']['imgs'], ops['end_points']['weighed_mask']]

        loss_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        outputs = sess.run(output_list + loss_list, feed_dict=feed_dict)

        _, step, lr_val, gt_rot_pnts_val, gt_ivts_xyz_val, gt_ivts_dist_val, gt_direction_val, \
        pred_xyz_val, pred_dist_val, pred_direction_val, sample_img_points_val, imgs_val, weighed_mask_val = outputs[:-len(losses)]

        for il, lossname in enumerate(losses.keys()):
            if lossname == "ivts_xyz_avg_diff":
                xyz_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_dist_avg_diff":
                dist_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_direction_avg_diff":
                direction_avg_diff_epoch += outputs[len(output_list)+il]
            losses[lossname] += outputs[len(output_list)+il]

        # outstr = "   "
        # for il, lossname in enumerate(losses.keys()):
        #         outstr += '%s: %f, ' % (lossname, outputs[len(output_list)+il])
        # # outstr += " weight mask =" + str(weighed_mask_val)
        # # outstr += " gt_ivts_xyz_val =" + str(batch_data['ivts'])
        # log_string(outstr)
                
        verbose_freq = 100.
        if (batch_idx + 1) % verbose_freq == 0:
            bid = 0
            # sampling
            if (batch_idx + 1) % (200*verbose_freq) == 0:
                saveimg = (imgs_val[bid, :, :, :] * 255).astype(np.uint8)
                samplept_img = sample_img_points_val[bid, ...]
                choice = np.random.randint(samplept_img.shape[0], size=100)
                samplept_img = samplept_img[choice, ...]
                for j in range(samplept_img.shape[0]):
                    x = int(samplept_img[j, 0])
                    y = int(samplept_img[j, 1])
                    cv2.circle(saveimg, (x, y), 3, (0, 0, 255, 255), -1)
                cv2.imwrite(os.path.join(RESULT_PATH, '%d_img_pnts.png' % batch_idx), saveimg)

                np.savetxt(os.path.join(RESULT_PATH, '%d_input_pnts.txt' % batch_idx), gt_rot_pnts_val[bid,:,:])

                np.savetxt(os.path.join(RESULT_PATH, '%d_ivts_pred.txt' % batch_idx), np.concatenate((gt_rot_pnts_val[bid,:,:] + pred_xyz_val[bid,:,:], np.expand_dims(pred_dist_val[bid,:,0],1)), axis=1))
                np.savetxt(os.path.join(RESULT_PATH, '%d_ivts_gt.txt' % batch_idx), np.concatenate((gt_rot_pnts_val[bid,:,:] + gt_ivts_xyz_val[bid,:,:], np.expand_dims(gt_ivts_dist_val[bid,:,0],1)), axis=1))

            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            outstr += "lr: %f" % (lr_val)
            outstr += ' time per b: %.02f, ' % ((time.time() - tic)/verbose_freq)
            outstr += ', fetch time per b: %.02f, ' % (fetch_time/verbose_freq)
            tic = time.time()
            fetch_time = 0
            log_string(outstr)
    print("avg xyz_avg_diff:", xyz_avg_diff_epoch / num_batches)
    print("avg dist_avg_diff:", dist_avg_diff_epoch / num_batches)
    print("avg direction_avg_diff:", direction_avg_diff_epoch / num_batches)
    return xyz_avg_diff_epoch / num_batches, dist_avg_diff_epoch / num_batches, direction_avg_diff_epoch / num_batches

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()






import numpy as np
import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing'))
import gpu_create_ivt as ct
import pymesh
import trimesh

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
	    area_sum+=area
	area_lst = np.asarray(area_lst)
	amount_lst = (area_lst * total / area_sum).astype(np.int32)
	points_all=np.zeros((0,3), dtype=np.float32)
	for i in range(amount_lst.shape[0]):
	    mesh = mesh_list[i]
	    # print("start sample surface of ", mesh.faces.shape[0])
	    points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
	    # print("end sample surface")
	    points_all = np.concatenate([points_all,points], axis=0)
	ori_mesh = pymesh.load_mesh(model_file)
	index = ori_mesh.faces.reshape(-1)
	tries = ori_mesh.vertices[index].reshape([-1,3,3])
	return points_all, tries


def unisample_pnts(res, num, model_file, threshold=0.1, stdratio=10, gpu=0):
	gt_pnts, tries = get_normalized_mesh(model_file)
	unigrid = get_unigrid(res)
	inds = np.random.choice(unigrid.shape[0], num*8)
	unigrid = unigrid[inds] 
	uni_ivts = ct.gpu_calculate_ivt(unigrid, tries, gpu)
	uni_dist = np.linalg.norm(uni_ivts, axis=1)
	ind = uni_dist <= threshold
	_, uni_place, std = cal_std_loc(unigrid[ind], uni_ivts[ind], stdratio)
	return uni_place, std, np.full(std.shape[0], 1/std.shape[0]), gt_pnts, tries


def cal_std_loc(pnts, ivts, stdratio, stdlwb=0.0, stdupb=0.1):
	loc = pnts + ivts
	dist = np.linalg.norm(ivts, axis=1)

	std = np.minimum(stdupb, np.maximum(stdlwb, dist / stdratio))
	return dist, loc, np.tile(np.expand_dims(std,axis=1),(1,3))

def sample_from_MM(mean_loc, std, weights, num, distr = "gaussian"):
	inds = np.random.choice(weights.shape[0], num, p=weights)
	sampled_pnts = np.zeros((num, 3))
	for i in range(num):
		ind = inds[i]
		if distr == "gaussian":
			loc = np.random.normal(mean_loc[ind],std[ind])
		else:
			loc = ball_sample(mean_loc[ind],std[ind])
		sampled_pnts[i,:] = loc
	return sampled_pnts

def ball_sample(xyzmean, radius):
    uvw = np.random.normal(np.array([0,0,0]), np.array([1,1,1]))
    e = np.random.exponential(0.5)
    denom = (e + np.dot(uvw,uvw))**0.5
    # print(radius,"as radius")
    # print(uvw,"as uvw")
    # print(denom,"as denome")
    # print(uvw, radius, uvw/denom, uvw/denom*radius," as uvw, radius,  uvw/denom, uvw/denom*radius")
    return xyzmean + uvw/denom*radius


def nearsample_pnts(locs, tries, stdratio=10, form="reverse", gpu=0, stdlwb = 0.04, stdupb = 0.1):
    surface_ivts = ct.gpu_calculate_ivt(locs, tries, gpu)
    dist, surface_place, std = cal_std_loc(locs, surface_ivts, stdratio, stdlwb=stdlwb, stdupb=stdupb)
    if form == "even":
        weights = np.full(std.shape[0], 1/std.shape[0])
    elif form == "reverse":
        weights = 1.0/ np.maximum(dist, 3e-3)
        weights = weights / np.sum(weights)

    return surface_place, std, weights

if __name__ == "__main__":

    # nohup python -u create_point_sdf_grid.py &> create_sdf.log &

    #  full set
    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    # if FLAGS.category != "all":
    #     cats = {
    #         FLAGS.category:cats[FLAGS.category]
    #     }

    # create_ivt(32768, 0.01, cats, raw_dirs,
    #            lst_dir, uni_ratio=0.2, normalize=True, version=1, skip_all_exist=True)
    rounds = 5
    res = 0.005
    nums = 8000
    num_ratio = 2
    stdratio = 2
    stdupb = [0.1, 0.1, 0.1, 0.03, 0]
    stdlwb = [0.08, 0.04, 0.02, 0.003, 0]
    distr="ball"
    form="reverse"

    # mean_loc, std, weights, gt_pc, tries = unisample_pnts(res, nums, "/ssd1/datasets/ShapeNet/ShapeNetCore_v1_norm/04530566/5d48d75153eb221b476c772fd813166d/pc_norm.obj", threshold=0.1, stdratio=2, gpu=0)
    mean_loc, std, weights, gt_pc, tries = unisample_pnts(res, nums, "/ssd1/datasets/ShapeNet/ShapeNetCore_v1_norm/03001627/17e916fc863540ee3def89b32cef8e45/pc_norm.obj", threshold=0.1, stdratio=2, gpu=0)
    pc = sample_from_MM(mean_loc, std, weights, nums, distr=distr)
    np.savetxt("gt_pc.txt", gt_pc, delimiter=';')
    np.savetxt("gt_pc.xyz", gt_pc, delimiter=' ')
    np.savetxt("uni_pc.txt", pc, delimiter=';')
    np.savetxt("uni_pc.xyz", pc, delimiter=' ')
    for i in range(rounds):
    	if i == (rounds-2):
    		distr = "gaussian"
    		form = "even"
    	else:
    		nums = nums * num_ratio
    	loc, std, weights = nearsample_pnts(pc, tries, stdratio=2, form=form, gpu=0, stdlwb = stdlwb[i], stdupb=stdupb[i])
    	pc = sample_from_MM(loc, std, weights, nums, distr=distr)
    	print("pc.shape", pc.shape, "loc.shape", loc.shape, "std.shape", std.shape, "weights.shape", weights.shape)
    	np.savetxt("surf_pc{}.txt".format(i), pc, delimiter=';')
    	np.savetxt("surf_pc{}.xyz".format(i), pc, delimiter=' ')

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
parser.add_argument('--encoder', type=str, default='vgg_16', help='encoder model: vgg_16, resnet_v1_50, resnet_v1_101, resnet_v2_50, resnet_v2_101')
parser.add_argument('--category', type=str, default="all", help='Which single class to train on [default: None]')
parser.add_argument('--log_dir', default='checkpoint', help='Log dir [default: log]')
parser.add_argument('--num_pnts', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--uni_num', type=int, default=512, help='Point Number [default: 2048]')
parser.add_argument('--num_classes', type=int, default=1024, help='vgg dim')
parser.add_argument("--beta1", type=float, dest="beta1", default=0.5, help="beta1 of adams")
# parser.add_argument('--sdf_points_num', type=int, default=32, help='Sample Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--img_h', type=int, default=137, help='Image Height')
parser.add_argument('--img_w', type=int, default=137, help='Image Width')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--restore_model', default='', help='restore_model') #checkpoint/sdf_2d3d_sdfbasic2_nowd
parser.add_argument('--restore_modelcnn', default='', help='restore_model')#../models/CNN/pretrained_model/vgg_16.ckpt
parser.add_argument('--wd', type=float, default=1e-5, help='Initial learning rate [default: 0.001]')

parser.add_argument('--train_lst_dir', default=lst_dir, help='train mesh data list')
parser.add_argument('--test_lst_dir', default=lst_dir, help='test mesh data list')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_type', type=str, default="ntanh")
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
parser.add_argument('--bn', action='store_true')
parser.add_argument('--lossw', nargs='+', action='append', default=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
parser.add_argument('--distlimit', nargs='+', action='store', type=str, default=[1.0, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01, -0.01])

FLAGS = parser.parse_args()
FLAGS.distlimit = [float(i) for i in FLAGS.distlimit]
print(FLAGS)


os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu

assert os.path.exists(FLAGS.log_dir), "{} does not exist".format(FLAGS.log_dir)

RESULT_PATH = os.path.join(FLAGS.log_dir, 'test_results')
if not os.path.exists(RESULT_PATH): os.mkdir(RESULT_PATH)


LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log_test.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

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
    test_lst = os.path.join(FLAGS.test_lst_dir, cat_id+"_test.lst")
    with open(test_lst, 'r') as f:
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

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        FLAGS.learning_rate,  # Base learning rate.
                        batch * FLAGS.batch_size,  # Current index into the dataset.
                        FLAGS.decay_step,          # Decay step.
                        FLAGS.decay_rate,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 1e-6, name='lr') # CLIP THE LEARNING RATE!
    return learning_rate


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

            print("--- Get model and loss")
            # Get model and loss

            end_points = model.get_model(input_pls, is_training_pl, bn=False, bn_decay=None, FLAGS=FLAGS)
            loss, end_points = model.get_loss(end_points, FLAGS=FLAGS)
            # tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator

            # Create a session
            gpu_options = tf.compat.v1.GPUOptions()#(per_process_gpu_memory_fraction=0.99)
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.compat.v1.Session(config=config)

            ##### all
            update_variables = [x for x in tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)]


            # Init variables
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)

            ######### Loading Checkpoint ###############
            # CNN(Pretrained from ImageNet)

            saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES) if
                                    ('lr' not in v.name) and ('batch' not in v.name)])
            ckptstate = tf.train.get_checkpoint_state(FLAGS.restore_model)

            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(FLAGS.restore_model, os.path.basename(ckptstate.model_checkpoint_path))
                saver.restore(sess, LOAD_MODEL_FILE)

                try:
                    # load_model(sess, LOAD_MODEL_FILE, ['sdfprediction/fold1', 'sdfprediction/fold2', FLAGS.encoder],
                    #            strict=True)
                    # # load_model(sess, LOAD_MODEL_FILE, ['sdfprediction','vgg_16'], strict=True)
                    # with NoStdStreams():
                    #     print("NoStdStreams-----------------------------")
                    print("Model loaded in file: %s" % LOAD_MODEL_FILE)
                except:
                    print("Fail to load overall modelfile: %s" % FLAGS.restore_model)
                    exit()
            ###########################################

            ops = {'input_pls': input_pls,
                   'is_training_pl': is_training_pl,
                   'loss': loss,
                   'end_points': end_points}

            TEST_DATASET.start()
            log_string('**** TESTING  ****')
            sys.stdout.flush()
            test_one_epoch(sess, ops)
            TEST_DATASET.shutdown()


# def pc_normalize(pc, centroid=None):

#     """ pc: NxC, return NxC """
#     l = pc.shape[0]

#     if centroid is None:
#         centroid = np.mean(pc, axis=0)

#     pc = pc - centroid
#     # m = np.max(pc, axis=0)
#     m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))

#     pc = pc / m

#     return pc

def test_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    num_batches = int(len(TEST_DATASET) / FLAGS.batch_size)

    print('num_batches', num_batches)

    log_string(str(datetime.now()))

    losses = {}
    for lossname in ops['end_points']['losses'].keys():
        losses[lossname] = 0

    tic = time.time()
    fetch_time = 0
    xyz_avg_diff_epoch = 0
    locnorm_avg_diff_epoch = 0
    locsqrnorm_avg_diff_epoch = 0
    dist_avg_diff_epoch = 0
    direction_avg_diff_epoch = 0
    direction_abs_avg_diff_epoch = 0
    lvlnum_epoch = 0

    xyz_lvl_diff_epoch = 0
    locnorm_lvl_diff_epoch = 0
    locsqrnorm_lvl_diff_epoch = 0
    dist_lvl_diff_epoch = 0
    direction_lvl_diff_epoch = 0
    direction_abs_lvl_diff_epoch = 0

    for batch_idx in range(num_batches):
        start_fetch_tic = time.time()
        batch_data = TEST_DATASET.fetch()
        fetch_time += (time.time() - start_fetch_tic)
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['input_pls']['pnts']: batch_data['pnts'],
                     ops['input_pls']['ivts']: batch_data['ivts'],
                     ops['input_pls']['imgs']: batch_data['imgs'],
                     ops['input_pls']['obj_rot_mats']: batch_data['obj_rot_mats'],
                     ops['input_pls']['trans_mats']: batch_data['trans_mats']}
        output_list = [ops['end_points']['pnts_rot'], ops['end_points']['gt_ivts_xyz'], ops['end_points']['gt_ivts_dist'], ops['end_points']['gt_ivts_direction'], ops['end_points']['pred_ivts_xyz'], ops['end_points']['pred_ivts_dist'],ops['end_points']['pred_ivts_direction'], ops['end_points']['sample_img_points'], ops['end_points']['imgs'], ops['end_points']['weighed_mask']]

        loss_list = []
        lvl_list = []
        for il, lossname in enumerate(losses.keys()):
            loss_list += [ops['end_points']['losses'][lossname]]

        for il, diffname in enumerate(ops['end_points']['lvl'].keys()):
            lvl_list += [ops['end_points']['lvl'][diffname]]

        outputs = sess.run(output_list + loss_list+lvl_list, feed_dict=feed_dict)

        gt_rot_pnts_val, gt_ivts_xyz_val, gt_ivts_dist_val, gt_direction_val, \
        pred_xyz_val, pred_dist_val, pred_direction_val, sample_img_points_val, imgs_val, weighed_mask_val = outputs[:-len(losses)-len(lvl_list)]

        for il, lossname in enumerate(losses.keys()):
            if lossname == "ivts_xyz_avg_diff":
                xyz_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_dist_avg_diff":
                dist_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_direction_avg_diff":
                direction_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_direction_abs_avg_diff":
                direction_abs_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_locnorm_avg_diff":
                locnorm_avg_diff_epoch += outputs[len(output_list)+il]
            if lossname == "ivts_locsqrnorm_avg_diff":
                locsqrnorm_avg_diff_epoch += outputs[len(output_list)+il]
            losses[lossname] += outputs[len(output_list)+il]

        for il, diffname in enumerate(ops['end_points']['lvl'].keys()):
            if diffname == "xyz_lvl_diff":
                xyz_lvl_diff_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "dist_lvl_diff":
                dist_lvl_diff_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "direction_lvl_diff":
                direction_lvl_diff_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "direction_abs_lvl_diff":
                direction_abs_lvl_diff_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "num":
                lvlnum_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "locnorm_lvl_diff":
                locnorm_lvl_diff_epoch += outputs[len(output_list)+len(loss_list)+il]
            if diffname == "locsqrnorm_lvl_diff":
                locsqrnorm_lvl_diff_epoch += outputs[len(output_list) + len(loss_list) + il]

                
        verbose_freq = 100.
        if (batch_idx + 1) % verbose_freq == 0:
            bid = 0
            # sampling
            outstr = ' -- %03d / %03d -- ' % (batch_idx+1, num_batches)
            for lossname in losses.keys():
                outstr += '%s: %f, ' % (lossname, losses[lossname] / verbose_freq)
                losses[lossname] = 0
            outstr += ' time per b: %.02f, ' % ((time.time() - tic)/verbose_freq)
            outstr += ' fetch time per b: %.02f, ' % (fetch_time/verbose_freq)
            tic = time.time()
            fetch_time = 0
            log_string(outstr)

        if batch_idx % 100 == 0:
            bid = 0
            saveimg = (imgs_val[bid, :, :, :] * 255).astype(np.uint8)
            samplept_img = sample_img_points_val[bid, ...]
            choice = np.random.randint(samplept_img.shape[0], size=100)
            samplept_img = samplept_img[choice, ...]
            for j in range(samplept_img.shape[0]):
                x = int(samplept_img[j, 0])
                y = int(samplept_img[j, 1])
                cv2.circle(saveimg, (x, y), 3, (0, 0, 255, 255), -1)
            cv2.imwrite(os.path.join(RESULT_PATH, '%d_img_pnts.png' % (batch_idx)), saveimg)

            np.savetxt(os.path.join(RESULT_PATH, '%d_input_pnts.txt' % (batch_idx)), gt_rot_pnts_val[bid, :, :], delimiter=';')

            np.savetxt(os.path.join(RESULT_PATH, '%d_ivts_pred.txt' % (batch_idx)), np.concatenate((gt_rot_pnts_val[bid, :, :] + pred_xyz_val[bid, :, :], np.expand_dims(pred_dist_val[bid, :, 0], 1)), axis=1), delimiter=';')
            np.savetxt(os.path.join(RESULT_PATH, '%d_ivts_gt.txt' % (batch_idx)), np.concatenate((gt_rot_pnts_val[bid, :, :] + gt_ivts_xyz_val[bid, :, :], np.expand_dims(gt_ivts_dist_val[bid, :, 0], 1)), axis=1), delimiter=';')

    if FLAGS.distlimit is not None:
        print(
            '{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^10s}{:^15s}{:^8s}'.format("upper", "lower", "locnorm", "locsqrnorm", "xyz", "dist", "drct", "drct_abs",
                                                                   "count"))
        for i in range(len(FLAGS.distlimit) // 2):
            upper = FLAGS.distlimit[i * 2]
            lower = FLAGS.distlimit[i * 2 + 1]
            count = max(1, int(lvlnum_epoch[i]))
            xyz = xyz_lvl_diff_epoch[i] / count
            locnorm = locnorm_lvl_diff_epoch[i] / count
            locsqrnorm = locsqrnorm_lvl_diff_epoch[i] / count
            dist = dist_lvl_diff_epoch[i] / count
            drct = direction_lvl_diff_epoch[i] / count
            drct_avg = direction_abs_lvl_diff_epoch[i] / count
            # print(upper, lower, xyz, dist, drct,drct_avg, count,xyz_lvl_diff_epoch.shape)
            print('{:^10.3f}{:^10.3f}{:^10.5f}{:^10.5f}{:^10.5f}{:^10.5f}{:^10.5f}{:^15.5f}{:^8d}'.format(upper, lower, locnorm, locsqrnorm, xyz, dist, drct, drct_avg, count))

    print("avg xyz_avg_diff:", xyz_avg_diff_epoch / num_batches)
    print("avg locnorm_avg_diff:", locnorm_avg_diff_epoch / num_batches)
    print("avg locsqrnorm_avg_diff:", locsqrnorm_avg_diff_epoch / num_batches)
    print("avg dist_avg_diff:", dist_avg_diff_epoch / num_batches)
    print("avg direction_avg_diff:", direction_avg_diff_epoch / num_batches)
    print("avg direction_abs_avg_diff:", direction_abs_avg_diff_epoch / num_batches)

    return xyz_avg_diff_epoch / num_batches, dist_avg_diff_epoch / num_batches, direction_avg_diff_epoch / num_batches

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    try:
        test()
    finally:
        print("finally")
        TEST_DATASET.shutdown()
        LOG_FOUT.close()

import create_file_lst
import h5py
import os
import numpy as np
import random
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import points_tries_dist_pycuda as ptdcuda
import argparse
import pymesh
import trimesh
import pandas as pd
from pyntcloud import PyntCloud

def save_norm(loc, norm, outfile):
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((loc, norm)),
        columns=["x", "y", "z", "nx", "ny", "nz"]))
    cloud.to_file(outfile)

START = 0
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
FLAGS=None

def create_h5_ivt_pt(gpu, cat_id, h5_file, verts, faces, surfpoints_sample, surfpoints, ungrid, norm_params, ivt_res, num_sample, uni_ratio):
    if faces.shape[0] > 2000000:
        print(cat_id,h5_file,"is too big!!! faces_size", faces.shape[0])
        return
    index = faces.reshape(-1)
    tries = verts[index].reshape([-1,3,3])
    print("tries.shape", tries.shape, faces.shape)
    ungrid = add_jitters(ungrid, std=0.005)
    surfpoints_sample = add_jitters(surfpoints_sample, std=0.05, type="uniform")
    uni_ivts = gpu_calculate_ivt(ungrid, tries,gpu)  # (N*8)x4 (x,y,z)
    surf_ivts = gpu_calculate_ivt(surfpoints_sample, tries, gpu)  # (N*8)x4 (x,y,z)
    print("start to write", h5_file)
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('uni_pnts', data=ungrid.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_pnts', data=surfpoints_sample.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('uni_ivts', data=uni_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_ivts', data=surf_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.close()


def create_ivt_obj(gpu, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, obj,
                   res, normalize, num_sample, cat_id, version, ungrid, uni_ratio, skip_all_exist):
    obj=obj.rstrip('\r\n')
    ivt_sub_dir = os.path.join(cat_sdf_dir, obj)
    norm_mesh_sub_dir = os.path.join(cat_norm_mesh_dir, obj)
    if not os.path.exists(ivt_sub_dir): os.makedirs(ivt_sub_dir)
    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    h5_file = os.path.join(ivt_sub_dir, "ivt_sample.h5")
    if  os.path.exists(h5_file) and skip_all_exist:
        print("skip existed: ", h5_file)
    else:
        if version == 1:
            model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
        else:
            model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        if normalize and (not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.obj")) or not os.path.exists(
                os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))):
            verts, faces, params, surfpoints = get_normalize_mesh(model_file, norm_mesh_sub_dir)
        else:
            verts, faces, surfpoints = get_mesh(norm_mesh_sub_dir)
            params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))

        surfpoints_sample = surfpoints[np.random.randint(surfpoints.shape[0], size = num_sample - int(uni_ratio*num_sample)),:] 
        create_h5_ivt_pt(gpu, cat_id, h5_file, verts, faces, surfpoints_sample, surfpoints, ungrid, params, res, num_sample, uni_ratio)

def create_pnt(num_sample, meshdir, cats, lst_dir, skip_all_exist=False):

    sdf_dir=raw_dirs["ivt_dir"]
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
    start=0
    unigrid = get_unigrid(res, int(uni_ratio*num_sample))
    thread_num=FLAGS.thread_num
    for catnm in cats.keys():
        cat_id = cats[catnm]
        cat_sdf_dir = os.path.join(sdf_dir, cat_id)
        if not os.path.exists(cat_sdf_dir): os.makedirs(cat_sdf_dir)
        cat_mesh_dir = os.path.join(raw_dirs["mesh_dir"], cat_id)
        cat_norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], cat_id)
        with open(lst_dir+"/"+str(cat_id)+"_test.lst", "r") as f:
            list_obj = f.readlines()
        with open(lst_dir+"/"+str(cat_id)+"_train.lst", "r") as f:
            list_obj += f.readlines()
        # print(list_obj)
        span = len(list_obj) // thread_num
        index = np.arange(len(list_obj))
        if FLAGS.shuffle: 
            np.random.shuffle(index)
        list_objs = [[list_obj[j] for j in index[i*span:min((i+1)*span,len(list_obj))].tolist()] for i in range(thread_num)]
        cat_mesh_dir_lst = [cat_mesh_dir for i in range(thread_num)]
        gpu_lst = [i % 4 for i in range(START, thread_num+START)]
        catnm_lst = [catnm for i in range(thread_num)]
        cat_norm_mesh_dir_lst = [cat_norm_mesh_dir for i in range(thread_num)]
        cat_sdf_dir_lst = [cat_sdf_dir for i in range(thread_num)]
        normalize_lst = [normalize for i in range(thread_num)]
        num_sample_lst = [num_sample for i in range(thread_num)]
        cat_id_lst = [cat_id for i in range(thread_num)]
        version_lst = [version for i in range(thread_num)]
        unigrid_lst = [unigrid for i in range(thread_num)]
        uni_ratio_lst = [uni_ratio for i in range(thread_num)]
        res_lst = [res for i in range(thread_num)]
        skip_all_exist_lst = [skip_all_exist for i in range(thread_num)]
        if thread_num > 1:
            with Parallel(n_jobs=thread_num) as parallel:
                vcts_part = parallel(delayed(create_ivt_distribute)
                    (gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_obj, res, normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist) for gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_obj, res, normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist in zip(gpu_lst, catnm_lst, cat_mesh_dir_lst, cat_norm_mesh_dir_lst, cat_sdf_dir_lst, list_objs, res_lst, normalize_lst, num_sample_lst, cat_id_lst, version_lst, unigrid_lst, uni_ratio_lst, skip_all_exist_lst))
        else:
            vcts_part = create_ivt_distribute(-1, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_objs[0], res, normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist)
    print("finish all")

def create_ivt_distribute(gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_obj, res, normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist):
    for i in range(len(list_obj)):
        create_ivt_obj(gpu, cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_obj[i],
            res, normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist)
        print("finish {}/{} for {}".format(i,len(list_obj),catnm))


# def create_single_pnt(mesh_dir, ref_dir, total, num):
#     model_file = os.path.join(mesh_dir, "pc_norm.obj")
#     print("trimesh_load:", model_file)
#     mesh_list = trimesh.load_mesh(model_file, process=False)
#     if not isinstance(mesh_list, list):
#         mesh_list = [mesh_list]
#     area_sum = 0
#     area_lst = []
#     for idx, mesh in enumerate(mesh_list):
#         area = np.sum(mesh.area_faces)
#         area_lst.append(area)
#         area_sum += area
#     area_lst = np.asarray(area_lst)
#     amount_lst = (area_lst * total / area_sum).astype(np.int32)
#     points_all = np.zeros((0, 3), dtype=np.float32)
#     for i in range(amount_lst.shape[0]):
#         mesh = mesh_list[i]
#         # print("start sample surface of ", mesh.faces.shape[0])
#         points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
#         # print("end sample surface")
#         points_all = np.concatenate([points_all, points], axis=0)
#     if os.path.exists(ref_dir):
#
#     choice = np.asarray(random.sample(range(points_all.shape[0]), num), dtype=np.int32)
#     points_all = points_all[choice]
#     surfpoints = np.random.shuffle(points_all)
#     return surfpoints


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_num', type=int, default='1', help='how many objs are creating at the same time')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--category', type=str, default="all",
                        help='Which single class to generate on [default: all, can be chair or plane, etc.]')
    FLAGS = parser.parse_args()

    # nohup python -u gpu_create_ivt.py &> create_sdf.log &

    #  full set
    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    if FLAGS.category != "all":
        cats = {
            FLAGS.category:cats[FLAGS.category]
        }

    create_single_pnt()

    # create_pnt(32768*2, meshdir, cats, lst_dir, uni_ratio=0.5, normalize=True, version=1, skip_all_exist=True)
    #
    # norm_mesh_sub_dir = "/ssd1/datasets/ShapeNet/ShapeNetCore_v1_norm/03001627/17e916fc863540ee3def89b32cef8e45"
    # verts, faces, surfpoints, facenorm, vertnorm = get_normal(norm_mesh_sub_dir)
    # print(np.linalg.norm(facenorm,axis=1))
    #
    # save_norm(surfpoints, facenorm, os.path.join("./", "surf_face_norm.ply"))
    # save_norm(surfpoints, vertnorm, os.path.join("./", "surf_vert_norm.ply"))
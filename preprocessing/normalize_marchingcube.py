import create_file_lst
import h5py
import os
import numpy as np
import pymesh
import random
from joblib import Parallel, delayed
import trimesh
from scipy.interpolate import RegularGridInterpolator
import time
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import points_tries_dist_pycuda as ptdcuda
import argparse


START = 0
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
FLAGS=None

def normalize_single_obj(obj_dir):
    source_obj = os.path.join(obj_dir,"17e916fc863540ee3def89b32cef8e45.obj")
    target_obj = os.path.join(obj_dir,"normalized_ori.obj")
    total = 16384 * 100
    print("trimesh_load:", source_obj)
    mesh_list = trimesh.load_mesh(source_obj, process=False)
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
        # print("end sample surface")
        points_all = np.concatenate([points_all, points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    # param_file = os.path.join(norm_mesh_sub_dir, "pc_norm.txt")
    ori_mesh = pymesh.load_mesh(source_obj)
    print("centroid, m", centroid, m)
    verts = (ori_mesh.vertices - centroid) / float(m)
    surfpoints = (points_all - centroid) / float(m)
    pymesh.save_mesh_raw(target_obj, verts, ori_mesh.faces);
    params = np.concatenate([centroid, np.expand_dims(m, axis=0)])
    # np.savetxt(param_file, params)
    print("export_mesh", target_obj)
    return verts, ori_mesh.faces, params, surfpoints


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
    # if FLAGS.category != "all":
    #     cats = {
    #         FLAGS.category:cats[FLAGS.category]
    #     }
    #
    # normalize_all(32768*2, 0.01, cats, raw_dirs,
    #            lst_dir, uni_ratio=0.5, normalize=True, version=1, skip_all_exist=True)
    normalize_single_obj("./")
    
    # tries = [[
    #     [1, 2, 3],
    #     [4, 6, 9],
    #     [12, 11, 9]],[
    #     [4, 6, 9],
    #     [1, 2, 3],
    #     [12, 11, 9]]
    # ]
    # planes = get_plane_abcd(np.array(tries))
    # print(planes)
    # planes[1,:]=np.array([2,-2,5,8])
    # dists = dist_2_plane(np.array([4,-4,3]), planes)
    # print(dists)

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
from normal_gen import save_norm
from sklearn.neighbors import DistanceMetric as dm
from sklearn.neighbors import NearestNeighbors
from random import sample
import h5py

lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

slim = tf.contrib.slim


def get_normalize_mesh(model_dir, norm_mesh_sub_dir, target_dir):
    norm_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    target_norm_file = os.path.join(target_dir, "pc_norm.obj")
    command_str = "cp " + norm_file + " " + target_norm_file
    print("command:", command_str)
    os.system(command_str)

    model_obj = os.path.join(model_dir, "model.obj")
    target_model_obj = os.path.join(target_dir, "model.obj")
    params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))
    print("trimesh_load:", model_obj)

    centroid = params[:3]
    m = params[3]
    print("centroid, m", centroid, m)

    ori_mesh = pymesh.load_mesh(model_obj)
    verts = (ori_mesh.vertices - centroid) / float(m)
    pymesh.save_mesh_raw(target_model_obj, verts, ori_mesh.faces)



def get_ivt_h5(ivt_h5_file, cat_id, obj):
    # print(ivt_h5_file)
    h5_f = h5py.File(ivt_h5_file, 'r')
    uni_pnts, surf_pnts, sphere_pnts, uni_ivts, surf_ivts, sphere_ivts, norm_params = None, None, None, None, None, None, None
    try:
        norm_params = h5_f['norm_params'][:].astype(np.float32)
        if 'uni_pnts' in h5_f.keys() and 'uni_ivts' in h5_f.keys():
            uni_pnts = h5_f['uni_pnts'][:].astype(np.float32)
            uni_ivts = h5_f['uni_ivts'][:].astype(np.float32)
            uni_onedge = h5_f['uni_onedge'][:].astype(np.float32)
        else:
            raise Exception(cat_id, obj, "no uni ivt and sample")
        if ('surf_pnts' in h5_f.keys() and 'surf_ivts' in h5_f.keys()):
            surf_pnts = h5_f['surf_pnts'][:].astype(np.float32)
            surf_ivts = h5_f['surf_ivts'][:].astype(np.float32)
            surf_onedge = h5_f['surf_onedge'][:].astype(np.float32)
        else:
            raise Exception(cat_id, obj, "no surf ivt and sample")
        if ('sphere_pnts' in h5_f.keys() and 'sphere_ivts' in h5_f.keys()):
            sphere_pnts = h5_f['sphere_pnts'][:].astype(np.float32)
            sphere_ivts = h5_f['sphere_ivts'][:].astype(np.float32)
            sphere_onedge = h5_f['sphere_onedge'][:].astype(np.float32)
        else:
            raise Exception(cat_id, obj, "no uni ivt and sample")
    finally:
        h5_f.close()
    return uni_pnts, surf_pnts, sphere_pnts, uni_ivts, surf_ivts, sphere_ivts, norm_params, uni_onedge, surf_onedge, sphere_onedge

def get_onedge(source, onedge):
    onedgeindex = np.argwhere(onedge > 0.5)[:,0]
    ontriindex = np.argwhere(onedge < 0.5)[:,0]
    print(onedge.shape,onedgeindex.shape,ontriindex.shape)
    return source[onedgeindex], source[ontriindex]


if __name__ == "__main__":

    catnm = "chair"
    cat = cats[catnm]
    obj = "4f42be4dd95e5e0c76e9713f57a5fcb6"
    target_dir = "test/check_align"
    os.makedirs(target_dir, exist_ok=True)

    model_dir = os.path.join(raw_dirs["mesh_dir"], cat, obj)
    norm_mesh_sub_dir = os.path.join(raw_dirs["real_norm_mesh_dir"], cat, obj)
    ivf_dir = os.path.join(raw_dirs["ivt_dir"], cat, obj)
    get_normalize_mesh(model_dir, norm_mesh_sub_dir, target_dir)


    ivf_file=os.path.join(ivf_dir, "ivt_sample.h5")
    uni_pnts, surf_pnts, sphere_pnts, uni_ivts, surf_ivts, sphere_ivts, norm_params, uni_onedge, surf_onedge, sphere_onedge = get_ivt_h5(ivf_file, cat, obj)

    save_norm(uni_pnts, uni_ivts, os.path.join(target_dir, "uni_l.ply"))
    save_norm(surf_pnts, surf_ivts, os.path.join(target_dir, "surf_l.ply"))
    save_norm(sphere_pnts, sphere_ivts, os.path.join(target_dir, "sphere_l.ply"))

    save_norm(uni_pnts+uni_ivts, uni_ivts, os.path.join(target_dir, "uni_s.ply"))
    save_norm(surf_pnts+surf_ivts, surf_ivts, os.path.join(target_dir, "surf_s.ply"))
    save_norm(sphere_pnts+sphere_ivts, sphere_ivts, os.path.join(target_dir, "sphere_s.ply"))

    uni_e_pnts, uni_t_pnts = get_onedge(uni_pnts, uni_onedge)
    uni_e_ivts, uni_t_ivts = get_onedge(uni_ivts, uni_onedge)
    surf_e_pnts, surf_t_pnts = get_onedge(surf_pnts, surf_onedge)
    surf_e_ivts, surf_t_ivts = get_onedge(surf_ivts, surf_onedge)
    sphere_e_pnts, sphere_t_pnts = get_onedge(sphere_pnts, sphere_onedge)
    sphere_e_ivts, sphere_t_ivts = get_onedge(sphere_ivts, sphere_onedge)

    save_norm(uni_e_pnts, uni_e_ivts, os.path.join(target_dir, "uni_e.ply"))
    save_norm(surf_e_pnts, surf_e_ivts, os.path.join(target_dir, "surf_e.ply"))
    save_norm(sphere_e_pnts, sphere_e_ivts, os.path.join(target_dir, "sphere_e.ply"))

    save_norm(uni_t_pnts, uni_t_ivts, os.path.join(target_dir, "uni_t.ply"))
    save_norm(surf_t_pnts, surf_t_ivts, os.path.join(target_dir, "surf_t.ply"))
    save_norm(sphere_t_pnts, sphere_t_ivts, os.path.join(target_dir, "sphere_t.ply"))

    print("done!")

# ./binvox -aw -dc -pb -d 1024 -fit pc_norm.obj

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
import utils.pointTriangleDistance as ptd
import argparse
import binvox_rw

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--thread_num', type=int, default='1', help='how many objs are creating at the same time')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
FLAGS = parser.parse_args()


def write_vox_h5(vox_file, h5_file, params):
    with open(vox_file, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        data_bool = model.data
        data_bool_bi = np.packbits(data_bool, axis=None)
        dims = np.array(model.dims, dtype=np.int)
        translate = np.array(model.translate, dtype=np.float32)
        scale = np.float32(model.scale)
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('dims', data=dims, compression='gzip', compression_opts=4)
    f1.create_dataset('translate', data=translate, compression='gzip', compression_opts=4)
    f1.create_dataset('scale', data=scale)
    f1.create_dataset('params', data=params, compression='gzip', compression_opts=4)
    f1.create_dataset('vox_bi', data=data_bool_bi, compression='gzip', compression_opts=4)
    f1.close()
    print("{} is written, dims={}".format(h5_file, dims))
    return data_bool


def read_vox_h5(h5_file):
    f1 = h5py.File(h5_file, 'r')
    print("f1[dims]", f1["dims"][:])
    print("f1[translate]", f1["translate"][:])
    print("f1[scale]", f1["scale"])
    print("f1[vox_bi]", f1["vox_bi"][:])
    data_binary = f1["vox_bi"][:]
    dims = f1["dims"][:]
    size = dims[0] * dims[1] * dims[2]
    data = np.unpackbits(data_binary, axis=None)[:size].reshape(dims).astype(np.bool)
    f1.close()
    return data


def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384 * 5
    print("trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)
    if not isinstance(mesh_list, list):
        mesh_list = [mesh_list]
    area_sum = 0
    area_lst = []
    meshes = []
    if isinstance(mesh_list[0], trimesh.Scene):
        for idx, mesh in enumerate(mesh_list):
            meshes += [trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()]
    else:
        meshes = mesh_list
    for idx, mesh in enumerate(meshes):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum += area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = meshes[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        # print("end sample surface")
        points_all = np.concatenate([points_all, points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    param_file = os.path.join(norm_mesh_sub_dir, "pc_norm.txt")
    ori_mesh = pymesh.load_mesh(model_file)
    print("centroid, m", centroid, m)
    verts = (ori_mesh.vertices - centroid) / float(m)
    surfpoints = (points_all - centroid) / float(m)
    pymesh.save_mesh_raw(obj_file, verts, ori_mesh.faces);
    params = np.concatenate([centroid, np.expand_dims(m, axis=0)])
    np.savetxt(param_file, params)
    print("export_mesh", obj_file, param_file)
    return verts, ori_mesh.faces, params, surfpoints


def create_vox_obj(binvox_command, cat_mesh_dir, cat_norm_mesh_dir, cat_vox_dir, obj, res,
                   cat_id, normalize, version, skip_all_exist):
    obj = obj.rstrip('\r\n')
    vox_sub_dir = os.path.join(cat_vox_dir, obj)
    norm_mesh_sub_dir = os.path.join(cat_norm_mesh_dir, obj)
    if not os.path.exists(vox_sub_dir): os.makedirs(vox_sub_dir)
    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    h5_file = os.path.join(vox_sub_dir, "vox.h5")
    vox_file = os.path.join(vox_sub_dir, "pc_norm.binvox")
    if os.path.exists(h5_file) and skip_all_exist:
        print("skip existed: ", h5_file)
    else:
        if os.path.exists(vox_file) and skip_all_exist:
            params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))
            print("skip create vox file: ", vox_file)
        else:
            if version == 1:
                model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
            else:
                model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
            if normalize and (not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.obj")) or not os.path.exists(
                    os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))):
                    _, _, params, _ = get_normalize_mesh(model_file, norm_mesh_sub_dir)
            else:
                params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))
            model_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
            vox_inter_file = os.path.join(norm_mesh_sub_dir, "pc_norm.binvox")
            # ./binvox -aw -dc -pb -d 1024 -fit pc_norm.obj
            command_str = binvox_command + " -aw -dc -pb -d " + str(res) + " -fit " + model_file
            print("command:", command_str)
            os.system(command_str)
            command_str2 = "mv " + vox_inter_file + " " + vox_file
            print("command:", command_str2)
            os.system(command_str2)
        write_vox_h5(vox_file, h5_file, params)


def create_vox(binvox_command, res, cats, raw_dirs, lst_dir, uni_ratio=0.2, normalize=True, version=1,
               skip_all_exist=False):
    vox_dir = raw_dirs["vox_dir_h5"]
    if not os.path.exists(vox_dir): os.makedirs(vox_dir)
    start = 0
    for catnm in cats.keys():
        cat_id = cats[catnm]
        cat_vox_dir = os.path.join(vox_dir, str(res), cat_id)
        if not os.path.exists(cat_vox_dir): os.makedirs(cat_vox_dir)
        cat_mesh_dir = os.path.join(raw_dirs["mesh_dir"], cat_id)
        cat_norm_mesh_dir = os.path.join(raw_dirs["norm_mesh_dir"], cat_id)
        with open(lst_dir + "/" + str(cat_id) + "_test.lst", "r") as f:
            list_obj = f.readlines()
        with open(lst_dir + "/" + str(cat_id) + "_train.lst", "r") as f:
            list_obj += f.readlines()
        # print(list_obj)
        repeat = len(list_obj)
        indx_lst = [i for i in range(start, start + repeat)]
        voxcommand_lst = [binvox_command for i in range(repeat)]
        cat_vox_dir_lst = [cat_vox_dir for i in range(repeat)]
        cat_norm_mesh_dir_lst = [cat_norm_mesh_dir for i in range(repeat)]
        cat_mesh_dir_lst = [cat_mesh_dir for i in range(repeat)]
        res_lst = [res for i in range(repeat)]
        cat_id_lst = [cat_id for i in range(repeat)]
        skip_all_exist_lst = [skip_all_exist for i in range(repeat)]
        version_lst = [version for i in range(repeat)]
        normalize_lst = [normalize for i in range(repeat)]
        with Parallel(n_jobs=FLAGS.thread_num) as parallel:
            parallel(delayed(create_vox_obj)
                     (binvox_command, cat_mesh_dir, cat_norm_mesh_dir, cat_vox_dir, obj, res,
                      cat_id, normalize, version, skip_all_exist)
                     for binvox_command, cat_mesh_dir, cat_norm_mesh_dir, cat_vox_dir, obj,
                         res, cat_id, normalize, version, skip_all_exist in
                     zip(voxcommand_lst,
                         cat_mesh_dir_lst,
                         cat_norm_mesh_dir_lst,
                         cat_vox_dir_lst,
                         list_obj,
                         res_lst, cat_id_lst, normalize_lst, version_lst, skip_all_exist_lst))
        start += repeat
    print("finish all")


if __name__ == "__main__":

    # nohup python -u create_vox.py --thread_num 18 &> create_vox.log &

    #  full set
    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    if FLAGS.category != "all":
        cats = {
            FLAGS.category: cats[FLAGS.category]
        }
    # cats = {"airplane": "02691156", "chair": "03001627"}
    create_vox("./binvox", 1024, cats, raw_dirs,
               lst_dir, uni_ratio=0.2, normalize=True, version=1, skip_all_exist=True)
    # data_bool = write_vox_h5("pc_norm.binvox","pc_norm.h5")
    # print(data_bool, np.sum(data_bool))
    # data_recon = read_vox_h5("pc_norm.h5")
    # print(np.array_equal(data_bool, data_recon), np.sum(data_recon))

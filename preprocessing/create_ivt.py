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

CUR_PATH = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--thread_num', type=int, default='9', help='how many objs are creating at the same time')
parser.add_argument('--category', type=str, default="all", help='Which single class to generate on [default: all, can '
                                                                'be chair or plane, etc.]')
FLAGS = parser.parse_args()

def get_unigrid(ivt_res, uni_num):
    grids = 1 / ivt_res 
    x = np.linspace(-1, 1, num=grids).astype(np.float32)
    y = np.linspace(-1, 1, num=grids).astype(np.float32)
    z = np.linspace(-1, 1, num=grids).astype(np.float32)
    choicex = np.array(random.sample(range(0, grids), uni_num))
    choicey = np.array(random.sample(range(0, grids), uni_num))
    choicez = np.array(random.sample(range(0, grids), uni_num))
    x_vals = x[choicex]
    y_vals = y[choicey]
    z_vals = z[choicez]
    return np.stack([x_vals, y_vals, z_vals], axis=0)

def add_jitters(uni_grid, uni_num, std=0.02):
    jitterx = np.random.normal(0, 0.02, 3 * uni_num).reshape([uni_num,3])
    return uni_grid + jitterx

def sample_ivt_uni(cat_id, num_sample, ivt_res):
    start = time.time()
    params = sdf_dict["param"]
    sdf_values = sdf_dict["value"].flatten()
    # print("np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values)",
    #       np.min(sdf_values), np.mean(sdf_values), np.max(sdf_values))
    x = np.linspace(params[0], params[3], num=ivt_res + 1).astype(np.float32)
    y = np.linspace(params[1], params[4], num=ivt_res + 1).astype(np.float32)
    z = np.linspace(params[2], params[5], num=ivt_res + 1).astype(np.float32)
    dis = sdf_values - iso_val
    sdf_pt_val = np.zeros((0,4), dtype=np.float32)
    for i in range(len(percentages)):
        ind = np.argwhere((dis >= percentages[i][0]) & (dis < percentages[i][1]))
        if len(ind) < percentages[i][2]:
            if i < len(percentages)-1:
                percentages[i+1][2] += percentages[i][2] - len(ind)
            percentages[i][2] = len(ind)
        if len(ind) == 0:
            print("len(ind) ==0 for cate i")
            continue
        choice = np.random.randint(len(ind), size=percentages[i][2])
        choosen_ind = ind[choice]
        x_ind = choosen_ind % (ivt_res + 1)
        y_ind = (choosen_ind // (ivt_res + 1)) % (ivt_res + 1)
        z_ind = choosen_ind // (ivt_res + 1) ** 2
        x_vals = x[x_ind]
        y_vals = y[y_ind]
        z_vals = z[z_ind]
        vals = sdf_values[choosen_ind]
        sdf_pt_val_bin = np.concatenate((x_vals, y_vals, z_vals, vals), axis = -1)
        # print("np.min(vals), np.mean(vals), np.max(vals)", np.min(vals), np.mean(vals), np.max(vals))
        print("sdf_pt_val_bin.shape", sdf_pt_val_bin.shape)
        sdf_pt_val = np.concatenate((sdf_pt_val, sdf_pt_val_bin), axis = 0)
    print("percentages", percentages)
    print("sample_sdf: {} s".format(time.time()-start))
    return sdf_pt_val, check_insideout(cat_id, sdf_values, sdf_res, x,y,z)

def check_insideout(cat_id, sdf_val, sdf_res, x, y, z):
    # "chair": "03001627",
    # "bench": "02828884",
    # "cabinet": "02933112",
    # "car": "02958343",
    # "airplane": "02691156",
    # "display": "03211117",
    # "lamp": "03636649",
    # "speaker": "03691459",
    # "rifle": "04090263",
    # "sofa": "04256520",
    # "table": "04379243",
    # "phone": "04401088",
    # "watercraft": "04530566"
    if cat_id in ["02958343", "02691156", "04530566"]:
        x_ind = np.argmin(np.absolute(x))
        y_ind = np.argmin(np.absolute(y))
        z_ind = np.argmin(np.absolute(z))
        all_val = sdf_val.flatten()
        num_val = all_val[x_ind+y_ind*(sdf_res+1)+z_ind*(sdf_res+1)**2]
        return num_val > 0.0
    else:
        return False

def create_h5_ivt_pt(cat_id, h5_file, verts, faces, points_uni, centroid, m, ivt_res, num_sample, normalize):
    print(verts.shape,faces.shape)
    index = faces.reshape(-1)
    tries = verts[index].reshape([-1,3,3])
    # print(tries[0], faces[0], verts[:3,:])
    sampleivt = sample_ivt_uni(cat_id, num_sample, ivt_res)  # (N*8)x4 (x,y,z)
    print("sampleivt", sampleivt.shape)
    print("start to write", h5_file)
    norm_params = np.concatenate((centroid, np.asarray([m]).astype(np.float32)))
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('pc_ivt_sample', data=sampleivt.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.close()



def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384
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
        print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    ori_mesh = pymesh.load_mesh(model_file)
    print("centroid, m", centroid, m)
    verts = (ori_mesh.vertices - centroid) / float(m)
    points_uni = (points_all - centroid) / float(m)
    pymesh.save_mesh_raw(obj_file, verts, ori_mesh.faces);
    print("export_mesh", obj_file)
    return verts, ori_mesh.faces, centroid, m, points_uni


def create_ivt_obj(cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, obj,
                   res, indx, normalize, num_sample, cat_id, version, ungrid, skip_all_exist):
    obj=obj.rstrip('\r\n')
    ivt_sub_dir = os.path.join(cat_sdf_dir, obj)
    norm_mesh_sub_dir = os.path.join(cat_norm_mesh_dir, obj)
    if not os.path.exists(ivt_sub_dir): os.makedirs(ivt_sub_dir)
    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    # sdf_file = os.path.join(sdf_sub_dir, "isosurf.sdf")
    # flag_file = os.path.join(sdf_sub_dir, "isinsideout.txt")
    # cube_obj_file = os.path.join(norm_mesh_sub_dir, "isosurf.obj")
    h5_file = os.path.join(ivt_sub_dir, "ivt_sample.h5")
    if  os.path.exists(h5_file) and skip_all_exist:
        print("skip existed: ", h5_file)
    else:
        if version == 1:
            model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
        else:
            model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        # try:
        if normalize:
            verts, faces, centroid, m, points_uni = get_normalize_mesh(model_file, norm_mesh_sub_dir)
        create_h5_ivt_pt(cat_id, h5_file, verts, faces, points_uni, 
            centroid, m, res, num_sample, normalize)
        # except:
        #     print("%%%%%%%%%%%%%%%%%%%%%%%% fail to process ", model_file)

def create_ivt(num_sample, res, cats, raw_dirs, lst_dir, normalize=True, version=1, skip_all_exist=False):

    sdf_dir=raw_dirs["ivt_dir"]
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
    start=0
    unigrid = get_unigrid(res)
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
        repeat = len(list_obj)
        indx_lst = [i for i in range(start, start+repeat)]
        cat_mesh_dir_lst=[cat_mesh_dir for i in range(repeat)]
        cat_norm_mesh_dir_lst=[cat_norm_mesh_dir for i in range(repeat)]
        cat_sdf_dir_lst=[cat_sdf_dir for i in range(repeat)]
        res_lst=[res for i in range(repeat)]
        normalize_lst=[normalize for i in range(repeat)]
        num_sample_lst=[num_sample for i in range(repeat)]
        cat_id_lst=[cat_id for i in range(repeat)]
        version_lst=[version for i in range(repeat)]
        unigrid_lst=[unigrid for i in range(repeat)]
        skip_all_exist_lst=[skip_all_exist for i in range(repeat)]
        # with Parallel(n_jobs=FLAGS.thread_num) as parallel:
        #     parallel(delayed(create_ivt_obj)
        #     (cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, obj, res,
        #      indx, norm, num_sample, cat_id,version, unigrid, skip_all_exist)
        #     for cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, obj,
        #         res, indx, norm, num_sample, cat_id,version, unigrid, skip_all_exist in
        #         zip(cat_mesh_dir_lst,
        #         cat_norm_mesh_dir_lst,
        #         cat_sdf_dir_lst,
        #         list_obj,
        #         res_lst, indx_lst, normalize_lst, num_sample_lst,
        #         cat_id_lst, version_lst, unigrid_lst, skip_all_exist_lst))
        create_ivt_obj(cat_mesh_dir_lst[0], cat_norm_mesh_dir_lst[0], cat_sdf_dir_lst[0], list_obj[0],
            res_lst[0], indx_lst[0], normalize_lst[0], num_sample_lst[0], cat_id_lst[0], version_lst[0], unigrid_lst[0], skip_all_exist_lst[0])
        start+=repeat
    print("finish all")


if __name__ == "__main__":

    # nohup python -u create_point_sdf_grid.py &> create_sdf.log &

    #  full set
    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    if FLAGS.category != "all":
        cats = {
            FLAGS.category:cats[FLAGS.category]
        }

    create_ivt(32768, 256, cats, raw_dirs,
               lst_dir, normalize=True, version=1, skip_all_exist=True)

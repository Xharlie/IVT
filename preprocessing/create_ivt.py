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
    grids = int(1 / ivt_res) 
    x = np.linspace(-1, 1, num=grids).astype(np.float32)
    y = np.linspace(-1, 1, num=grids).astype(np.float32)
    z = np.linspace(-1, 1, num=grids).astype(np.float32)
    choicex = np.random.randint(grids, size = uni_num)
    choicey = np.random.randint(grids, size = uni_num)
    choicez = np.random.randint(grids, size = uni_num)
    x_vals = x[choicex]
    y_vals = y[choicey]
    z_vals = z[choicez]
    return np.stack([x_vals, y_vals, z_vals], axis=1)

def add_jitters(uni_grid, std=0.05):
    jitterx = np.random.normal(0, std, 3 * uni_grid.shape[0]).reshape([uni_grid.shape[0],3])
    print(uni_grid.shape, jitterx.shape)
    return uni_grid + jitterx

def thresh_edge_tries(tries, edge_thresh=0.02):
    triesAB = np.linalg.norm(tries[:,1,:] - tries[:,0,:], axis = 1)
    triesAC = np.linalg.norm(tries[:,2,:] - tries[:,0,:], axis = 1)
    triesBC = np.linalg.norm(tries[:,2,:] - tries[:,1,:], axis = 1)
    edgetries = triesAB + triesAC + triesBC
    if np.amax(edgetries) <= edge_thresh:
        print("np.amax(edgetries) <= edge_thresh")
        return None 
    largetries = tries[edgetries>edge_thresh]
    return largetries

def rank_dist_tries(points, tries, rank_thresh=100):
    avg_points = np.mean(tries, axis=1)
    span = 5000 * 26500 // points.shape[0]
    ind_lst=[]
    points_tries = np.tile(np.expand_dims(points, axis=1),(1,span,1))
    part_thresh = rank_thresh
    if tries.shape[0] > 800000:
        print("tries.shape[0] > 800000:",tries.shape[0])
        part_thresh = rank_thresh // 10
    for i in range(avg_points.shape[0]//span+1):
        avg_point = avg_points[span*i:min((i+1)*span,avg_points.shape[0])]
        if avg_point.shape[0] != span:
            points_tries = np.tile(np.expand_dims(points, axis=1),(1,avg_point.shape[0],1))
        dist = np.linalg.norm(points_tries - np.expand_dims(avg_point, axis=0), axis=2)
        if dist.shape[1] > part_thresh:
            ind_close = np.argpartition(dist, part_thresh)
            ind_close = ind_close[:,:part_thresh]
        else:
            ind_close = np.tile(np.expand_dims(np.arange(dist.shape[1]), axis=0),(dist.shape[0],1))
        # print(ind_close.shape)
        ind_close = ind_close+span*i
        ind_lst.append(ind_close)
    inds = np.concatenate(ind_lst,axis=1)    
    close_tries = tries[inds]
    avg_point = np.mean(close_tries, axis=2)
    points_tries = np.tile(np.expand_dims(points, axis=1),(1,close_tries.shape[1],1))
    dist = np.linalg.norm(points_tries - avg_point, axis=2)
    if dist.shape[1] > rank_thresh:
        ind_close = np.argpartition(dist, rank_thresh)
        ind_close = ind_close[:,:rank_thresh]
    else:
        ind_close = np.tile(np.expand_dims(np.arange(dist.shape[1]), axis=0),(dist.shape[0],1))
    first_index = np.tile(np.expand_dims(np.arange(ind_close.shape[0]), axis=1),(1,rank_thresh))
    close_tries=close_tries[first_index,ind_close]
    return close_tries

def calculate_ivt(points, tries):
    start = time.time()
    rank_thresh = 100
    num_tries = tries.shape[0]
    abcd = None
    e = None
    if tries.shape[0] > 2000:
        print("tries.shape[0] > 2000:", tries.shape[0])
        large_tries = thresh_edge_tries(tries, edge_thresh=0.24)
        close_tries = rank_dist_tries(points, tries, rank_thresh=rank_thresh)
        print("close_tries.shape", close_tries.shape)
        num_tries = close_tries.shape[1]
        if large_tries is not None:
            num_tries += large_tries.shape[0]
            abcd = get_plane_abcd(large_tries)
            e = np.linalg.norm(abcd[:,:3], axis = 1)

    print("points.shape[0]", points.shape[0])
    times = points.shape[0] * num_tries // (25000 * 6553 * 2) + 1
    span = points.shape[0] // times + 1
    print("times:",times,"span:",span)
    vcts = []
    for i in range(times):
        smindx = i * span
        lgindx = min(points.shape[0], (i+1) * span)
        print("smindx",smindx,"lgindx",lgindx)
        if tries.shape[0] <= 2000:
            print("tries.shape[0] <= 2000:", tries.shape[0])
            tries_lst = [tries for i in range(lgindx-smindx)]
        elif large_tries is not None:
            large_tries_tile =np.tile(large_tries, (lgindx-smindx,1,1,1))
            print("large_tries_tile is not None, shape", large_tries_tile.shape)
            candid_tries = np.concatenate([close_tries[smindx:lgindx], large_tries_tile], axis=1)
            tries_lst = [candid_tries[i] for i in range(lgindx-smindx)]
        else: 
            tries_lst = [close_tries[i] for i in range(smindx, lgindx)]
        abcd_lst = [abcd for i in range(lgindx-smindx)]
        e_lst = [e for i in range(lgindx-smindx)]
        point_lst = [points[i] for i in range(smindx, lgindx)]
        # print("len(point_lst)", len(point_lst), point_lst[0].shape, len(tries_lst), tries_lst[0].shape)
        with Parallel(n_jobs=18) as parallel:
            vcts_part = parallel(delayed(calculate_ivt_single)
                (abcd, e, point, tries) for abcd, e, point, tries in zip(abcd_lst, e_lst, point_lst, tries_lst))
        # print("len(vcts_part):", len(vcts_part))
        vcts += vcts_part
    print("time diff:", time.time() - start)
    vcts = np.stack(vcts, axis=0)
    print(vcts.shape)
    return vcts

def get_plane_abcd(tries):
    v1 = tries[:,2,:] - tries[:,0,:]
    v2 = tries[:,1,:] - tries[:,0,:]
    cp = np.cross(v1,v2)
    d = np.sum(np.multiply(cp, tries[:,2,:]), axis=1)
    abcd = np.concatenate([cp, np.expand_dims(-d,axis=1)], axis = 1)
    print("abcd.shape", abcd.shape)
    return abcd

def dist_2_plane(point, planes, e):
    point1=np.concatenate([point, np.array([1])])
    point = np.tile(np.expand_dims(point1, axis=0),(planes.shape[0],1))
    d = abs(np.sum(np.multiply(point, planes), axis=1))
    return d/e


def calculate_ivt_single(planes, e, point, tries):
    minimum = 3
    plane_check_start_index = tries.shape[0]
    if planes is not None:
        plane_check_start_index = tries.shape[0] - planes.shape[0]
        plane_dists = dist_2_plane(point, planes, e)
    vct_shortest = np.zeros([3])
    count = -1
    for tri in tries:
        count+=1
        if count >= plane_check_start_index:
            if plane_dists[count - plane_check_start_index] > minimum:
                # print("count,{},plane_check_start_index,{},plane_dists[count - plane_check_start_index],{},minimum,{}".format(count,plane_check_start_index,plane_dists[count - plane_check_start_index],minimum))
                continue
        dist, vct = ptd.pointTriangleDistance(tri, point) 
        if dist < minimum:
            vct_shortest = vct
            minimum = dist
    # print("start, points {} in {}".format(i, points.shape[0]))
    return vct_shortest



# def check_insideout(cat_id, sdf_val, sdf_res, x, y, z):
#     # "chair": "03001627",
#     # "bench": "02828884",
#     # "cabinet": "02933112",
#     # "car": "02958343",
#     # "airplane": "02691156",
#     # "display": "03211117",
#     # "lamp": "03636649",
#     # "speaker": "03691459",
#     # "rifle": "04090263",
#     # "sofa": "04256520",
#     # "table": "04379243",
#     # "phone": "04401088",
#     # "watercraft": "04530566"
#     if cat_id in ["02958343", "02691156", "04530566"]:
#         x_ind = np.argmin(np.absolute(x))
#         y_ind = np.argmin(np.absolute(y))
#         z_ind = np.argmin(np.absolute(z))
#         all_val = sdf_val.flatten()
#         num_val = all_val[x_ind+y_ind*(sdf_res+1)+z_ind*(sdf_res+1)**2]
#         return num_val > 0.0
#     else:
#         return False


def create_h5_ivt_pt(cat_id, h5_file, verts, faces, surfpoints_sample, surfpoints, ungrid, norm_params, ivt_res, num_sample, uni_ratio):
    index = faces.reshape(-1)
    tries = verts[index].reshape([-1,3,3])
    print("tries.shape", tries.shape, faces.shape)
    ungrid = add_jitters(ungrid, std=0.05)
    surfpoints_sample = add_jitters(surfpoints_sample, std=0.05)
    uni_ivts = calculate_ivt(ungrid, tries)  # (N*8)x4 (x,y,z)
    surf_ivts = calculate_ivt(surfpoints_sample, tries)  # (N*8)x4 (x,y,z)
    print("start to write", h5_file)
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('uni_pnts', data=ungrid.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_pnts', data=surfpoints_sample.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('uni_ivts', data=uni_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_ivts', data=surf_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.close()



def get_normalize_mesh(model_file, norm_mesh_sub_dir):
    total = 16384 * 5 * 2
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
    print("export_mesh", obj_file)
    return verts, ori_mesh.faces, params, surfpoints

def get_mesh(norm_mesh_sub_dir):
    total = 16384 * 5 * 2
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    print("trimesh_load:", obj_file)
    mesh_list = trimesh.load_mesh(obj_file, process=False)
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
    ori_mesh = pymesh.load_mesh(obj_file)
    return ori_mesh.vertices, ori_mesh.faces, points_all

def create_ivt_obj(cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, obj,
                   res, indx, normalize, num_sample, cat_id, version, ungrid, uni_ratio, skip_all_exist):
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
        create_h5_ivt_pt(cat_id, h5_file, verts, faces, surfpoints_sample, surfpoints, ungrid, params, res, num_sample, uni_ratio)

def create_ivt(num_sample, res, cats, raw_dirs, lst_dir, uni_ratio=0.2, normalize=True, version=1, skip_all_exist=False):

    sdf_dir=raw_dirs["ivt_dir"]
    if not os.path.exists(sdf_dir): os.makedirs(sdf_dir)
    start=0
    unigrid = get_unigrid(res, int(uni_ratio*num_sample))
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
        
        for i in range(repeat):
            create_ivt_obj(cat_mesh_dir, cat_norm_mesh_dir, cat_sdf_dir, list_obj[i],
                res, indx_lst[i], normalize, num_sample, cat_id, version, unigrid, uni_ratio, skip_all_exist)
            print("finish {}/{} for {}".format(i,repeat,catnm))
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

    create_ivt(32768, 0.01, cats, raw_dirs,
               lst_dir, uni_ratio=0.2, normalize=True, version=1, skip_all_exist=True)
    
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
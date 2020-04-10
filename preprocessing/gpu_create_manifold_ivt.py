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


def get_unigrid(ivt_res):
    grids = int(2 / ivt_res)
    x = np.linspace(-1, 1, num=grids).astype(np.float32)
    y = np.linspace(-1, 1, num=grids).astype(np.float32)
    z = np.linspace(-1, 1, num=grids).astype(np.float32)
    return np.stack([x, y, z], axis=1)

def get_ballgrid(angles_num):
    phi = np.linspace(0, np.pi, num=angles_num)
    theta = np.linspace(0, 2 * np.pi, num=2 * angles_num)
    x = np.outer(np.sin(theta), np.cos(phi)).reshape(-1)
    y = np.outer(np.sin(theta), np.sin(phi)).reshape(-1)
    z = np.outer(np.cos(theta), np.ones_like(phi)).reshape(-1)
    return np.stack([x, y, z], axis=1)

def sample_uni(xyz, uni_num):
    choicex = np.random.randint(xyz.shape[0], size=uni_num)
    choicey = np.random.randint(xyz.shape[0], size=uni_num)
    choicez = np.random.randint(xyz.shape[0], size=uni_num)
    x_vals = xyz[choicex,0]
    y_vals = xyz[choicey,1]
    z_vals = xyz[choicez,2]
    return np.stack([x_vals, y_vals, z_vals], axis=1)

def sample_balluni(xyz, uni_num):
    choice = np.random.randint(xyz.shape[0], size=uni_num)
    return xyz[choice]

def add_jitters(points, std=0.05, type="uniform"):
    if type == "normal":
        jitterx = np.random.normal(0, std, 3 * points.shape[0]).reshape([points.shape[0],3])
    else:
        jitterx = np.random.uniform(-std, std, 3 * points.shape[0]).reshape([points.shape[0],3])
    print(points.shape, jitterx.shape)
    return points + jitterx

def add_normal_jitters(points, normals, height=0.1, span=0.05):
    jitterx = np.random.uniform(-height, height, points.shape[0]).reshape([points.shape[0],1])
    jitterx = np.multiply(jitterx, normals) + np.random.uniform(-span, span, 3*points.shape[0]).reshape([points.shape[0],3])
    print(points.shape, jitterx.shape)
    return points + jitterx

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

def gpu_calculate_ivt(points, tries, gpu, from_marchingcube):
    start = time.time()
    num_tries = tries.shape[0]
    vcts = []
    if from_marchingcube:
        ind_start = time.time()
        avg_points = np.mean(tries, axis=1)
        topk_ind = ptdcuda.cal_topkind(points, avg_points, gpu=gpu)
        topk_tries = np.take(tries, topk_ind, axis=0)
        print("finish, pick top5_ind", topk_ind.shape, topk_tries.shape, "time diff:", time.time() - ind_start)
        ivtround_start = time.time()
        ivt, dist = ptdcuda.pnts_tries_ivts(points, None, topk_tries=topk_tries, gpu=gpu)
        print("finish ptdcuda ivt, dist:", ivt.shape, dist.shape, "time diff:", time.time() - ivtround_start)
        vcts_part = ptdcuda.closet(ivt, dist)
        vcts.append(vcts_part)
    else:
        times = points.shape[0] * num_tries // (25000 * 6553 * 3) + 1
        span = points.shape[0] // times + 1
        vcts = []
        for i in range(times):
            print("start ptdcuda: {}/{}".format(i + 1, times))
            smindx = i * span
            lgindx = min(points.shape[0], (i+1) * span)
            pnts = points[smindx:lgindx,:]
            ivtround_start = time.time()
            ivt, dist = ptdcuda.pnts_tries_ivts(pnts, tries, gpu=gpu)
            print("finish, ivt, dist", ivt.shape, dist.shape, "time diff:", time.time() - ivtround_start)
            vcts_part = ptdcuda.closet(ivt, dist)
            vcts.append(vcts_part)
            print("end ptdcuda: {}/{}".format(i+1, times))
    ivt_closest = vcts[0] if len(vcts) == 0 else np.concatenate(vcts, axis=0)
    print("times", times, "ivt_closest.shape", ivt_closest.shape, "time diff:", time.time() - start)
    return ivt_closest

# def closest_dist(points, tries, gpu):
#     min_dists = ptdcuda.cal_dist(points, avg_points, gpu=gpu)
#     return min_dists


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


def create_h5_ivt_pt(gpu, cat_id, h5_file, verts, faces, surfpoints_sample, surfnormals_sample, ball_samples, ungridsamples, norm_params, ivt_res, num_sample, uni_ratio, from_marchingcube):
    if faces.shape[0] > 2000000:
        print(cat_id,h5_file,"is too big!!! faces_size", faces.shape[0])
        return
    index = faces.reshape(-1)
    tries = verts[index].reshape([-1,3,3])
    print("tries.shape", tries.shape, faces.shape)
    ball_samples = add_jitters(ball_samples, std=0.01, type="uniform")
    ungridsamples = add_jitters(ungridsamples, std=0.005, type="uniform")
    surfpoints_sample = add_normal_jitters(surfpoints_sample, surfnormals_sample, height=0.1)
    sphere_ivts = gpu_calculate_ivt(ball_samples, tries,gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
    uni_ivts = gpu_calculate_ivt(ungridsamples, tries,gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
    surf_ivts = gpu_calculate_ivt(surfpoints_sample, tries, gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
    print("start to write", h5_file)
    f1 = h5py.File(h5_file, 'w')
    f1.create_dataset('uni_pnts', data=ungridsamples.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('sphere_pnts', data=ball_samples.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_pnts', data=surfpoints_sample.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('uni_ivts', data=uni_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('sphere_ivts', data=sphere_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('surf_ivts', data=surf_ivts.astype(np.float32), compression='gzip', compression_opts=4)
    # f1.create_dataset('surf_normals', data=surfnormals_sample.astype(np.float32), compression='gzip', compression_opts=4)
    f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
    f1.close()
    # np.savetxt(h5_file[:-3]+"ball_samples.txt",ball_samples, delimiter=';')
    # np.savetxt(h5_file[:-3]+"ungridsamples.txt",ungridsamples, delimiter=';')
    # np.savetxt(h5_file[:-3]+"surfpoints_sample.txt",surfpoints_sample, delimiter=';')



def get_normalize_mesh(model_file, norm_mesh_sub_dir, ref_sub_dir, pntnum):
    total = 16384 * 50
    print("trimesh_load:", model_file)
    ref_file = os.path.join(ref_sub_dir, "isosurf.obj")
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
    face_norm_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        if not os.path.exists(ref_file):
            face_norms = find_normal(index, mesh)
            face_norm_all = np.concatenate([face_norm_all, face_norms], axis=0)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    param_file = os.path.join(norm_mesh_sub_dir, "pc_norm.txt")
    pnt_file = os.path.join(norm_mesh_sub_dir, "pnt_{}.txt".format(pntnum))
    params = np.concatenate([centroid, np.expand_dims(m, axis=0)])
    np.savetxt(param_file, params)
    print("export_mesh", obj_file)
    from_marchingcube = False
    if not os.path.exists(ref_file):
        ori_mesh = pymesh.load_mesh(model_file)
        print("centroid, m", centroid, m)
    else:
        from_marchingcube = True
        mesh_list = trimesh.load_mesh(ref_file, process=False)
        print("trimesh_load ref_file:", ref_file)
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
            face_norms = find_normal(index, mesh)
            face_norm_all = np.concatenate([face_norm_all, face_norms], axis=0)
            # print("end sample surface")
            points_all = np.concatenate([points_all, points], axis=0)
        centroid = np.mean(points_all, axis=0)
        points_all = points_all - centroid
        m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
        ori_mesh = pymesh.load_mesh(ref_file)
    surfpoints = points_all / float(m)
    print("centroid, m", centroid, m)
    verts = (ori_mesh.vertices - centroid) / float(m)
    pymesh.save_mesh_raw(obj_file, verts, ori_mesh.faces)
    pntchoice = np.random.randint(surfpoints.shape[0], size=pntnum)
    np.savetxt(pnt_file, np.concatenate([surfpoints[pntchoice], face_norm_all[pntchoice]], axis=1), delimiter=';')
    print("export_pntnorm", pnt_file)
    return verts, ori_mesh.faces, params, surfpoints, face_norm_all, from_marchingcube

def find_normal(index, mesh):
    all_face_normals = mesh.face_normals
    face_norms = all_face_normals[index]
    return face_norms

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
    face_norm_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        face_norms = find_normal(index, mesh)
        face_norm_all = np.concatenate([face_norm_all, face_norms], axis=0)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    ori_mesh = pymesh.load_mesh(obj_file)
    return ori_mesh.vertices, ori_mesh.faces, points_all, face_norm_all

def create_ivt_obj(gpu, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, obj, res, normalize, num_sample, pntnum, cat_id, version, ungrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist):
    obj=obj.rstrip('\r\n')
    ivt_sub_dir = os.path.join(cat_ivt_dir, obj)
    ref_sub_dir = os.path.join(cat_ref_dir, obj)
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
        if normalize and (not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.obj")) or not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))):
            verts, faces, params, surfpoints, surfnormals, from_marchingcube = get_normalize_mesh(model_file, norm_mesh_sub_dir, ref_sub_dir, pntnum)
        else:
            verts, faces, surfpoints, surfnormals = get_mesh(norm_mesh_sub_dir)
            params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))
        surfchoice = np.random.randint(surfpoints.shape[0], size = int(surf_ratio*num_sample))
        surfpoints_sample = surfpoints[surfchoice,:]
        surfnormals_sample = surfnormals[surfchoice,:]
        ungridsamples = sample_uni(ungrid, int(uni_ratio*num_sample))
        ball_samples = sample_balluni(ballgrid, int((1.0-uni_ratio-surf_ratio)*num_sample))
        create_h5_ivt_pt(gpu, cat_id, h5_file, verts, faces, surfpoints_sample, surfnormals_sample, ball_samples, ungridsamples, params, res, num_sample, uni_ratio, from_marchingcube)

def create_ivt(num_sample, pntnum, res, angles_num, cats, raw_dirs, lst_dir, uni_ratio=0.3, surf_ratio=0.4, normalize=True, version=1, skip_all_exist=False):

    ivt_dir=raw_dirs["ivt_mani_dir"]
    ref_dir=raw_dirs["ref_mani_dir"]
    if not os.path.exists(ivt_dir): os.makedirs(ivt_dir)
    start=0
    unigrid = get_unigrid(res)
    ballgrid = get_ballgrid(angles_num)
    thread_num=FLAGS.thread_num
    for catnm in cats.keys():
        cat_id = cats[catnm]
        cat_ivt_dir = os.path.join(ivt_dir, cat_id)
        cat_ref_dir = os.path.join(ref_dir, cat_id)
        if not os.path.exists(cat_ivt_dir): os.makedirs(cat_ivt_dir)
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
        cat_ivt_dir_lst = [cat_ivt_dir for i in range(thread_num)]
        normalize_lst = [normalize for i in range(thread_num)]
        num_sample_lst = [num_sample for i in range(thread_num)]
        pntnum_lst = [pntnum for i in range(thread_num)]
        cat_id_lst = [cat_id for i in range(thread_num)]
        version_lst = [version for i in range(thread_num)]
        unigrid_lst = [unigrid for i in range(thread_num)]
        ballgrid_lst = [ballgrid for i in range(thread_num)]
        uni_ratio_lst = [uni_ratio for i in range(thread_num)]
        surf_ratio_lst = [surf_ratio for i in range(thread_num)]
        res_lst = [res for i in range(thread_num)]
        cat_ref_dir_lst = [cat_ref_dir for i in range(thread_num)]
        skip_all_exist_lst = [skip_all_exist for i in range(thread_num)]
        if thread_num > 1:
            with Parallel(n_jobs=thread_num) as parallel:
                vcts_part = parallel(delayed(create_ivt_distribute)
                    (gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist) for gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist in zip(gpu_lst, catnm_lst, cat_mesh_dir_lst, cat_norm_mesh_dir_lst, cat_ivt_dir_lst, cat_ref_dir_lst, list_objs, res_lst, normalize_lst, num_sample_lst, pntnum_lst, cat_id_lst, version_lst, unigrid_lst, ballgrid_lst, uni_ratio_lst, surf_ratio_lst, skip_all_exist_lst))
        else:
            vcts_part = create_ivt_distribute(-1, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, list_objs[0], res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist)
    print("finish all")

def create_ivt_distribute(gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist):
    for i in range(len(list_obj)):
        create_ivt_obj(gpu, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_ref_dir, list_obj[i],
            res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist)
        print("finish {}/{} for {}".format(i,len(list_obj),catnm))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_num', type=int, default='1', help='how many objs are creating at the same time')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--category', type=str, default="all",
                        help='Which single class to generate on [default: all, can be chair or plane, etc.]')
    FLAGS = parser.parse_args()

    # nohup python -u gpu_create_manifold_ivt.py --thread_num 3 --shuffle --category up &> create_ivt.log &

    #  full set
    lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    if FLAGS.category == "up":
        cats = {
            "chair": "03001627",
            "airplane": "02691156",
            "watercraft": "04530566",
            "rifle": "04090263",
            "display": "03211117",
            "lamp": "03636649"
        }
    elif FLAGS.category == "lower":
        cats = {
            "speaker": "03691459",
            "cabinet": "02933112",
            "bench": "02828884",
            "car": "02958343",
            "sofa": "04256520",
            "table": "04379243",
            "phone": "04401088"
        }
    elif FLAGS.category != "all":
        cats = {
            FLAGS.category:cats[FLAGS.category]
        }

    create_ivt(32768*3, 8192, 0.01, 100, cats, raw_dirs, lst_dir, uni_ratio=0.3, surf_ratio=0.4, normalize=True, version=1, skip_all_exist=True)

    # unigrid = get_unigrid(0.01)
    # ballgrid = get_ballgrid(100)
    #
    # print("unigrid.shape, ballgrid.shape", unigrid.shape, ballgrid.shape)
    #
    # create_ivt_obj(0, "/ssd1/datasets/ShapeNet/ShapeNetCore.v1/03001627", "./test/1/", "./test/1/", "/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/03001627", "17e916fc863540ee3def89b32cef8e45", 0.01, True, 32768*3, 8192, "03001627", 1, unigrid, ballgrid, 0.3, 0.4, True)
    #
    # create_ivt_obj(1, "/ssd1/datasets/ShapeNet/ShapeNetCore.v1/02958343",
    #                "./test/2/", "./test/2/",
    #                "/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/02958343", "1a7125aefa9af6b6597505fd7d99b613", 0.01, True,
    #                32768 * 3, 8192, "02958343", 1, unigrid, ballgrid, 0.3, 0.4, True)

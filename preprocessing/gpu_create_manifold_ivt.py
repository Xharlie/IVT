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
import normal_gen
from normal_gen import save_norm


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
    # jitterx = np.multiply(jitterx, normals) + np.random.uniform(-span, span, 3*points.shape[0]).reshape([points.shape[0],3])
    R = normal_gen.norm_z_matrix(normals, rect=False)
    round_points = np.matmul(R, np.expand_dims(round_sample(span,points.shape[0]),axis=2))
    print("jitterx.shape, R.shape, round_points.shape", jitterx.shape, R.shape, round_points.shape)
    jitterx = np.multiply(jitterx, normals) + np.squeeze(round_points)
    return points + jitterx

def round_sample(radius, num):
    angle = np.random.uniform(0,1, size=num) * 2 * np.pi
    r = np.sqrt(np.random.uniform(0,1, size=num)) * radius
    return np.stack([np.multiply(r, np.cos(angle)), np.multiply(r, np.sin(angle)), np.zeros_like(r)], axis=1)

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
    closest_ind = np.zeros((0),dtype=np.int)
    closest_onedge = np.zeros((0),dtype=np.float)
    if from_marchingcube:
        ind_start = time.time()
        avg_points = np.mean(tries, axis=1)
        topk_ind = ptdcuda.cal_topkind(points, avg_points, gpu=gpu)
        topk_tries = np.take(tries, topk_ind, axis=0)
        print("finish, pick topk_ind", topk_ind.shape, topk_tries.shape, "time diff:", time.time() - ind_start)
        ivtround_start = time.time()
        ivt, dist, on_edge = ptdcuda.pnts_tries_ivts(points, None, topk_tries=topk_tries, gpu=gpu)
        print("finish ptdcuda ivt, dist:", ivt.shape, dist.shape, "time diff:", time.time() - ivtround_start)
        vcts_part, minind, closest_onedge = ptdcuda.closet(ivt, dist, on_edge)
        minind = np.arange(minind.shape[0]) * 10 + minind
        closest_ind = np.take(topk_ind, minind)
        # print("closest_ind,minind",closest_ind.shape,minind.shape,minind)
        vcts.append(vcts_part)
    else:
        times = points.shape[0] * num_tries // (25000 * 6553 * 3) + 1
        span = points.shape[0] // times + 1
        vcts = []
        print("points.shape, tries.shape", points.shape, tries.shape)
        for i in range(times):
            print("start ptdcuda: {}/{}".format(i + 1, times))
            smindx = i * span
            lgindx = min(points.shape[0], (i+1) * span)
            pnts = points[smindx:lgindx,:]
            ivtround_start = time.time()
            ivt, dist, on_edge = ptdcuda.pnts_tries_ivts(pnts, tries, gpu=gpu)
            print("finish, ivt, dist", ivt.shape, dist.shape, "time diff:", time.time() - ivtround_start)
            vcts_part, minind, on_edge_closest = ptdcuda.closet(ivt, dist, on_edge)
            closest_ind = np.concatenate([closest_ind, minind],axis=0)
            closest_onedge = np.concatenate([closest_onedge, on_edge_closest],axis=0)
            vcts.append(vcts_part)
            print("end ptdcuda: {}/{}".format(i+1, times))
    ivt_closest = vcts[0] if len(vcts) == 0 else np.concatenate(vcts, axis=0)
    print("ivt_closest.shape", ivt_closest.shape, "time diff:", time.time() - start)
    return ivt_closest, closest_onedge, closest_ind


def gpu_calculate_ivt_norm(points, tries, face_norms, vert_norms, gpu, from_marchingcube):
    ivt_closest, on_edge_closest, closest_ind = gpu_calculate_ivt(points, tries, gpu, from_marchingcube)
    # print("uniq face ind",np.unique(closest_ind))
    tries = tries[closest_ind]
    face_norms = face_norms[closest_ind]
    vert_norms = vert_norms[closest_ind]
    print("gpu_calculate_ivt_norm : face_norms.shape, vert_norms.shape",face_norms.shape, vert_norms.shape)
    points_surf = points + ivt_closest
    inter_norm = normal_gen.interp(points_surf, face_norms, tries, vert_norms)
    return ivt_closest, on_edge_closest, inter_norm



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


def create_h5_ivt_pt(gpu, cat_id, h5_file, tries, face_norms, vert_norms, surfpoints_sample, surfnormals_sample, ball_samples, ungridsamples, norm_params, from_marchingcube, normalgt):
    if tries.shape[0] > 2000000:
        print(cat_id,h5_file,"is too big!!! faces_size", faces.shape[0])
        return
    elif tries.shape[0] > 35000:
        from_marchingcube = True
    print("tries.shape", tries.shape)
    ball_samples = add_jitters(ball_samples, std=0.01, type="uniform")
    ungridsamples = add_jitters(ungridsamples, std=0.005, type="uniform")
    surfpoints_sample = add_normal_jitters(surfpoints_sample, surfnormals_sample, height=0.1)

    if not normalgt:
        sphere_ivts, sphere_onedge, _ = gpu_calculate_ivt(ball_samples, tries,gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
        uni_ivts, uni_onedge, __ = gpu_calculate_ivt(ungridsamples, tries,gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
        surf_ivts, surf_onedge, __ = gpu_calculate_ivt(surfpoints_sample, tries, gpu,from_marchingcube)  # (N*8)x4 (x,y,z)
        print("save sphere_onedge.shape, uni_onedge.shape, surf_onedge.shape", sphere_onedge.shape, uni_onedge.shape, surf_onedge.shape)
    else:
        sphere_ivts, sphere_onedge, sphere_norm = gpu_calculate_ivt_norm(ball_samples, tries, face_norms, vert_norms, gpu, from_marchingcube)  # (N*8)x4 (x,y,z)
        uni_ivts, uni_onedge, uni_norm = gpu_calculate_ivt_norm(ungridsamples, tries, face_norms, vert_norms, gpu, from_marchingcube)  # (N*8)x4 (x,y,z)
        surf_ivts, surf_onedge, surf_norm = gpu_calculate_ivt_norm(surfpoints_sample, tries, face_norms, vert_norms, gpu, from_marchingcube)  # (N*8)x4 (x,y,z)
    print("start to write", h5_file)
    with h5py.File(h5_file, 'w') as f1:
        f1.create_dataset('uni_pnts', data=ungridsamples.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('sphere_pnts', data=ball_samples.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('surf_pnts', data=surfpoints_sample.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('uni_ivts', data=uni_ivts.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('sphere_ivts', data=sphere_ivts.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('surf_ivts', data=surf_ivts.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('uni_onedge', data=uni_onedge.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('sphere_onedge', data=sphere_onedge.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('surf_onedge', data=surf_onedge.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('norm_params', data=norm_params, compression='gzip', compression_opts=4)
        if normalgt:
            f1.create_dataset('uni_norm', data=uni_norm.astype(np.float32), compression='gzip', compression_opts=4)
            f1.create_dataset('sphere_norm', data=sphere_norm.astype(np.float32), compression='gzip', compression_opts=4)
            f1.create_dataset('surf_norm', data=surf_norm.astype(np.float32), compression='gzip', compression_opts=4)
    # np.savetxt(h5_file[:-3]+"ball_samples.txt",ball_samples, delimiter=';')
    # np.savetxt(h5_file[:-3]+"ungridsamples.txt",ungridsamples, delimiter=';')
    # np.savetxt(h5_file[:-3]+"surfpoints_sample.txt",surfpoints_sample, delimiter=';')


def get_mesh(norm_mesh_sub_dir, ref_sub_dir, pnt_dir, pntnum):
    ref_file = os.path.join(ref_sub_dir, "isosurf.obj")
    from_marchingcube = False
    if os.path.exists(ref_file):
        from_marchingcube = True
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
    all_face_normals = np.zeros((0, 3), dtype=np.float32)
    all_vert_normals = np.zeros((0, 3, 3), dtype=np.float32)
    all_tries = np.zeros((0, 3, 3), dtype=np.float32)
    sample_indices = np.zeros((0), dtype=np.int)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        sample_indices = np.concatenate([sample_indices, sample_indices.shape[0] + index], axis=0)
        vert_ind = mesh.faces.reshape(-1)
        all_tries = np.concatenate([all_tries, mesh.vertices[vert_ind].reshape([-1, 3, 3])], axis=0)
        all_face_normals = np.concatenate([all_face_normals, mesh.face_normals], axis=0)
        all_vert_normals = np.concatenate([all_vert_normals, mesh.vertex_normals[vert_ind].reshape([-1, 3, 3])], axis=0)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    face_norm_surfpnt = save_surface(pntnum, sample_indices, points_all, all_tries, all_face_normals, all_vert_normals, pnt_dir)
    return all_tries, all_face_normals, all_vert_normals, points_all, face_norm_surfpnt, from_marchingcube

def create_ivt_obj(gpu, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, obj, res, normalize, num_sample, pntnum, cat_id, version, ungrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel):
    obj=obj.rstrip('\r\n')
    ivt_sub_dir = os.path.join(cat_ivt_dir, obj)
    ref_sub_dir = os.path.join(cat_ref_dir, obj)
    norm_mesh_sub_dir = os.path.join(cat_norm_mesh_dir, obj)
    pnt_dir = os.path.join(cat_pnt_dir, obj)
    if not os.path.exists(ivt_sub_dir): os.makedirs(ivt_sub_dir)
    if not os.path.exists(norm_mesh_sub_dir): os.makedirs(norm_mesh_sub_dir)
    h5_file = os.path.join(ivt_sub_dir, "ivt_sample.h5")
    if os.path.exists(h5_file) and skip_all_exist:
        print("skip existed: ", h5_file)
    else:
        if version == 1:
            model_file = os.path.join(cat_mesh_dir, obj, "model.obj")
        else:
            model_file = os.path.join(cat_mesh_dir, obj, "models", "model_normalized.obj")
        if realmodel or normalize and (not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.obj")) or not os.path.exists(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))):
            if realmodel:
                all_tries, all_face_normals, all_vert_normals, params, surfpoints, surfnormals, from_marchingcube = get_normalize_mesh_real(model_file, norm_mesh_sub_dir, pnt_dir, ref_sub_dir, pntnum)
            else:
                all_tries, all_face_normals, all_vert_normals, params, surfpoints, surfnormals, from_marchingcube = get_normalize_mesh(model_file, norm_mesh_sub_dir, pnt_dir, ref_sub_dir, pntnum)
        else:
            all_tries, all_face_normals, all_vert_normals, surfpoints, surfnormals, from_marchingcube = get_mesh(norm_mesh_sub_dir, ref_sub_dir, pnt_dir, pntnum)
            from_marchingcube = os.path.exists(os.path.join(ref_sub_dir, "isosurf.obj"))
            params = np.loadtxt(os.path.join(norm_mesh_sub_dir, "pc_norm.txt"))
        surfchoice = np.random.randint(surfpoints.shape[0], size = int(surf_ratio*num_sample))
        surfpoints_sample = surfpoints[surfchoice,:]
        surfnormals_sample = surfnormals[surfchoice,:]
        ungridsamples = sample_uni(ungrid, int(uni_ratio*num_sample))
        ball_samples = sample_balluni(ballgrid, int((1.0-uni_ratio-surf_ratio)*num_sample))
        create_h5_ivt_pt(gpu, cat_id, h5_file, all_tries, all_face_normals, all_vert_normals, surfpoints_sample, surfnormals_sample, ball_samples, ungridsamples, params, from_marchingcube, normalgt)


def create_ivt(num_sample, pntnum, res, angles_num, cats, raw_dirs, lst_dir, uni_ratio=0.3, surf_ratio=0.4, normalize=True, version=1, skip_all_exist=False, normalgt=True, realmodel=False):

    if realmodel:
        norm_mesh_dir = raw_dirs["real_norm_mesh_dir"]
        ivt_dir = raw_dirs["ivt_dir"]
    else:
        norm_mesh_dir = raw_dirs["norm_mesh_dir"]
        ivt_dir = raw_dirs["ivt_mani_dir"]
    ref_dir=raw_dirs["ref_mani_dir"]
    if not os.path.exists(ivt_dir): os.makedirs(ivt_dir)
    if not os.path.exists(raw_dirs["pnt_dir"]): os.makedirs(raw_dirs["pnt_dir"])
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
        cat_pnt_dir = os.path.join(raw_dirs["pnt_dir"], cat_id)
        if not os.path.exists(cat_pnt_dir): os.makedirs(cat_pnt_dir)
        cat_norm_mesh_dir = os.path.join(norm_mesh_dir, cat_id)
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
        cat_pnt_dir_lst = [cat_pnt_dir for i in range(thread_num)]
        gpu_lst = [i % 4 for i in range(START, thread_num+START)]
        catnm_lst = [catnm for i in range(thread_num)]
        cat_norm_mesh_dir_lst = [cat_norm_mesh_dir for i in range(thread_num)]
        cat_ivt_dir_lst = [cat_ivt_dir for i in range(thread_num)]
        normalize_lst = [normalize for i in range(thread_num)]
        num_sample_lst = [num_sample for i in range(thread_num)]
        normalgt_lst = [normalgt for i in range(thread_num)]
        pntnum_lst = [pntnum for i in range(thread_num)]
        cat_id_lst = [cat_id for i in range(thread_num)]
        version_lst = [version for i in range(thread_num)]
        unigrid_lst = [unigrid for i in range(thread_num)]
        ballgrid_lst = [ballgrid for i in range(thread_num)]
        uni_ratio_lst = [uni_ratio for i in range(thread_num)]
        surf_ratio_lst = [surf_ratio for i in range(thread_num)]
        res_lst = [res for i in range(thread_num)]
        cat_ref_dir_lst = [cat_ref_dir for i in range(thread_num)]
        realmodel_lst = [realmodel for i in range(thread_num)]
        skip_all_exist_lst = [skip_all_exist for i in range(thread_num)]
        if thread_num > 1:
            with Parallel(n_jobs=thread_num) as parallel:
                vcts_part = parallel(delayed(create_ivt_distribute)
                    (gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel) for gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel in zip(gpu_lst, catnm_lst, cat_mesh_dir_lst, cat_norm_mesh_dir_lst, cat_ivt_dir_lst, cat_pnt_dir_lst, cat_ref_dir_lst, list_objs, res_lst, normalize_lst, num_sample_lst, pntnum_lst, cat_id_lst, version_lst, unigrid_lst, ballgrid_lst, uni_ratio_lst, surf_ratio_lst, skip_all_exist_lst,normalgt_lst, realmodel_lst))
        else:
            vcts_part = create_ivt_distribute(-1, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, list_objs[0], res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel)
    print("finish all")

def create_ivt_distribute(gpu, catnm, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, list_obj, res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel):
    for i in range(len(list_obj)):
        create_ivt_obj(gpu%4, cat_mesh_dir, cat_norm_mesh_dir, cat_ivt_dir, cat_pnt_dir, cat_ref_dir, list_obj[i],
            res, normalize, num_sample, pntnum, cat_id, version, unigrid, ballgrid, uni_ratio, surf_ratio, skip_all_exist, normalgt, realmodel)
        print("finish {}/{} for {}".format(i,len(list_obj),catnm))

def get_normalize_mesh_real(model_file, norm_mesh_sub_dir, pnt_dir, ref_sub_dir, pntnum):
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
    all_face_normals = np.zeros((0, 3), dtype=np.float32)
    all_vert_normals = np.zeros((0, 3, 3), dtype=np.float32)
    all_tries = np.zeros((0, 3, 3), dtype=np.float32)
    sample_indices = np.zeros((0), dtype=np.int)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        if not os.path.exists(ref_file):
            sample_indices = np.concatenate([sample_indices, sample_indices.shape[0] + index], axis=0)
        vert_ind = mesh.faces.reshape(-1)
        all_tries = np.concatenate([all_tries, mesh.vertices[vert_ind].reshape([-1, 3, 3])], axis=0)
        all_face_normals = np.concatenate([all_face_normals, mesh.face_normals], axis=0)
        all_vert_normals = np.concatenate([all_vert_normals, mesh.vertex_normals[vert_ind].reshape([-1, 3, 3])], axis=0)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    all_tries = (all_tries - centroid) / float(m)
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    param_file = os.path.join(norm_mesh_sub_dir, "pc_norm.txt")
    params = np.concatenate([centroid, np.expand_dims(m, axis=0)])
    np.savetxt(param_file, params)
    print("export_mesh", obj_file)
    from_marchingcube = False
    ori_mesh = pymesh.load_mesh(model_file)
    verts = (ori_mesh.vertices - centroid) / float(m)
    pymesh.save_mesh_raw(obj_file, verts, ori_mesh.faces)
    if not os.path.exists(ref_file):
        print("centroid, m", centroid, m)
        ref_face_normals = all_face_normals
    else:
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
        ref_face_normals = np.zeros((0, 3), dtype=np.float32)
        for i in range(amount_lst.shape[0]):
            mesh = mesh_list[i]
            points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
            sample_indices = np.concatenate([sample_indices, sample_indices.shape[0] + index], axis=0)
            ref_face_normals = np.concatenate([ref_face_normals, mesh.face_normals], axis=0)
            points_all = np.concatenate([points_all, points], axis=0)
        centroid = np.mean(points_all, axis=0)
        points_all = points_all - centroid
        m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    surfpoints = points_all / float(m)
    face_norm_surfpnt = ref_face_normals[sample_indices]
    return all_tries, all_face_normals, all_vert_normals, params, surfpoints, face_norm_surfpnt, from_marchingcube

def get_normalize_mesh(model_file, norm_mesh_sub_dir, pnt_dir, ref_sub_dir, pntnum):
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
    all_face_normals = np.zeros((0, 3), dtype=np.float32)
    all_vert_normals = np.zeros((0, 3, 3), dtype=np.float32)
    all_tries = np.zeros((0, 3, 3), dtype=np.float32)
    sample_indices = np.zeros((0), dtype=np.int)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        if not os.path.exists(ref_file):
            sample_indices = np.concatenate([sample_indices, sample_indices.shape[0] + index], axis=0)
            vert_ind = mesh.faces.reshape(-1)
            all_tries = np.concatenate([all_tries, mesh.vertices[vert_ind].reshape([-1, 3, 3])], axis=0)
            all_face_normals = np.concatenate([all_face_normals, mesh.face_normals], axis=0)
            all_vert_normals = np.concatenate([all_vert_normals, mesh.vertex_normals[vert_ind].reshape([-1, 3, 3])], axis=0)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    param_file = os.path.join(norm_mesh_sub_dir, "pc_norm.txt")
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
            sample_indices = np.concatenate([sample_indices, sample_indices.shape[0] + index], axis=0)
            vert_ind = mesh.faces.reshape(-1)
            all_tries = np.concatenate([all_tries, mesh.vertices[vert_ind].reshape([-1, 3, 3])], axis=0)
            all_face_normals = np.concatenate([all_face_normals, mesh.face_normals], axis=0)
            all_vert_normals = np.concatenate([all_vert_normals, mesh.vertex_normals[vert_ind].reshape([-1, 3, 3])], axis=0)
            # print("end sample surface")
            points_all = np.concatenate([points_all, points], axis=0)
        centroid = np.mean(points_all, axis=0)
        points_all = points_all - centroid
        m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
        ori_mesh = pymesh.load_mesh(ref_file)
    surfpoints = points_all / float(m)
    all_tries = (all_tries - centroid[np.newaxis, np.newaxis, :]) / float(m)
    print("centroid, m", centroid, m)
    verts = (ori_mesh.vertices - centroid) / float(m)
    pymesh.save_mesh_raw(obj_file, verts, ori_mesh.faces)
    face_norm_surfpnt = save_surface(pntnum, sample_indices, surfpoints, all_tries, all_face_normals, all_vert_normals, pnt_dir)
    return all_tries, all_face_normals, all_vert_normals, params, surfpoints, face_norm_surfpnt, from_marchingcube

def save_surface(pntnum, sample_indices, surfpoints, all_tries, all_face_normals, all_vert_normals, pnt_dir):
    os.makedirs(pnt_dir, exist_ok=True)
    pnt_file = os.path.join(pnt_dir, "pnt_{}.h5".format(pntnum))
    pntchoice = np.random.randint(surfpoints.shape[0], size=pntnum)
    sampled_surfpoints = surfpoints[pntchoice]
    face_tries = all_tries[sample_indices]
    face_norm_surfpnt = all_face_normals[sample_indices]
    vert_norms = all_vert_normals[sample_indices]
    sampled_face_norm_surfpnt = normal_gen.interp(sampled_surfpoints, face_norm_surfpnt[pntchoice], face_tries[pntchoice], vert_norms[pntchoice])
    with h5py.File(pnt_file, 'w') as f1:
        f1.create_dataset('pnt', data=sampled_surfpoints.astype(np.float32), compression='gzip', compression_opts=4)
        f1.create_dataset('normal', data=sampled_face_norm_surfpnt.astype(np.float32), compression='gzip', compression_opts=4)
    # np.savetxt(os.path.join(pnt_dir, "pnt_{}.txt".format(pntnum)), np.concatenate([sampled_surfpoints, sampled_face_norm_surfpnt], axis=1), delimiter=";")
    print("export_pntnorm", pnt_file)
    return face_norm_surfpnt

def test_h5(ivt_dir, pnt_dir, mesh_dir, target_dir):
    command_str = "cp " + mesh_dir + "pc_norm.obj " +target_dir
    print("command:", command_str)
    os.system(command_str)
    with h5py.File(os.path.join(ivt_dir, "ivt_sample.h5"),"r") as f1:
        uni_pnts = f1['uni_pnts'][:]
        sphere_pnts = f1['sphere_pnts'][:]
        surf_pnts = f1['surf_pnts'][:]
        uni_ivts = f1['uni_ivts'][:]
        sphere_ivts = f1['sphere_ivts'][:]
        surf_ivts = f1['surf_ivts'][:]
        uni_norm = f1['uni_norm'][:]
        sphere_norm = f1['sphere_norm'][:]
        surf_norm = f1['surf_norm'][:]
    with h5py.File(os.path.join(pnt_dir, "pnt_163840.h5"),"r") as f2:
        pnts = f2['pnt'][:]
        norms = f2['normal'][:]
    save_norm(uni_pnts+uni_ivts, uni_norm, os.path.join(target_dir, "uni.ply"))
    save_norm(sphere_pnts+sphere_ivts, sphere_norm, os.path.join(target_dir, "sphere.ply"))
    save_norm(surf_pnts+surf_ivts, surf_norm, os.path.join(target_dir, "surf.ply"))
    save_norm(pnts, norms, os.path.join(target_dir, "pnt.ply"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_num', type=int, default='1', help='how many objs are creating at the same time')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--realmodel', action='store_true')
    parser.add_argument('--category', type=str, default="all",
                        help='Which single class to generate on [default: all, can be chair or plane, etc.]')
    FLAGS = parser.parse_args()

    # nohup python -u gpu_create_manifold_ivt.py --thread_num 12 --shuffle --category all &> create_ivt.log &
    # nohup python -u gpu_create_manifold_ivt.py --thread_num 3 --shuffle --category chair --realmodel &> create_ivt.log &

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

    # get_mesh("/hdd_extra1/datasets/ShapeNet/ShapeNetCore_v1_norm/03001627/17e916fc863540ee3def89b32cef8e45", "/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/03001627/17e916fc863540ee3def89b32cef8e45", "./test/2/", 50000)

    create_ivt(32768*3, 163840, 0.01, 200, cats, raw_dirs, lst_dir, uni_ratio=0.4, surf_ratio=0.4, normalize=True, version=1, skip_all_exist=True, normalgt=False, realmodel=FLAGS.realmodel)




    # ivt_dir, pnt_dir, mesh_dir, target_dir = "/ssd1/datasets/ShapeNet/IVT_mani_v1/03001627/99ee0185fa2492859be1712545749b62/", "/ssd1/datasets/ShapeNet/pnt_163840/03001627/99ee0185fa2492859be1712545749b62/", "/hdd_extra1/datasets/ShapeNet/ShapeNetCore_v1_norm/03001627/99ee0185fa2492859be1712545749b62/","./test/2/"
    # test_h5(ivt_dir, pnt_dir, mesh_dir, target_dir)



    # unigrid = get_unigrid(0.01)
    # ballgrid = get_ballgrid(100)
    #
    # print("unigrid.shape, ballgrid.shape", unigrid.shape, ballgrid.shape)
    #
    # create_ivt_obj(0, "/ssd1/datasets/ShapeNet/ShapeNetCore.v1/03001627", "./test/1/", "./test/1/", "/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/03001627", "17e916fc863540ee3def89b32cef8e45", 0.01, True, 32768*3, 163840, "03001627", 1, unigrid, ballgrid, 0.3, 0.4, True)

    # create_ivt_obj(1, "/ssd1/datasets/ShapeNet/ShapeNetCore.v1/02958343",
    #                "./test/2/", "./test/2/",
    #                "/hdd_extra1/datasets/ShapeNet/march_cube_objs_v1/02958343", "1a7125aefa9af6b6597505fd7d99b613", 0.01, True,
    #                32768 * 3, 8192, "02958343", 1, unigrid, ballgrid, 0.3, 0.4, True)

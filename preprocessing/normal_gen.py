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

def norm_z_matrix(norm, rect=True):
    bs = norm.shape[0]
    z_b = np.repeat(np.array([(0., 0., 1.)]), bs, axis=0)
    print("norm.shape, z_b.shape",norm.shape, z_b.shape)
    normal = np.cross(norm, z_b) if rect else np.cross(z_b, norm)
    print("normal of norm rotate:", normal.shape)
    sinal = np.linalg.norm(normal,axis=1,keepdims=True)
    cosal = np.sum(np.multiply(norm, z_b),axis=1,keepdims=True)
    # sinalsqrrev = 1/np.square(np.maximum(1e-5, sinal))
    ux = normal[:,[0]]
    uy = normal[:,[1]]
    uz = normal[:,[2]]
    zr = np.zeros_like(ux)
    W = np.concatenate([zr, -uz, uy, uz, zr, -ux, -uy, ux, zr], axis=1).reshape((-1,3,3))
    I = np.repeat(np.expand_dims(np.identity(3), axis = 0), bs, axis=0)
    Wsqr = np.matmul(W, W)
    C = 1/np.maximum(1e-5,1+cosal)
    R = I + W + np.expand_dims(C, axis=2) * Wsqr
    return R


def get_normal(norm_mesh_sub_dir):
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
    face_norm_all=np.zeros((0,3), dtype=np.float32)
    vert_norm_all=np.zeros((0,3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        face_norms, vert_norm = find_normal(points,index,mesh)
        # print("end sample surface")
        points_all = np.concatenate([points_all,points], axis=0)
        face_norm_all = np.concatenate([face_norm_all,face_norms], axis=0)
        vert_norm_all = np.concatenate([vert_norm_all,vert_norm], axis=0)
    ori_mesh = pymesh.load_mesh(obj_file)
    return ori_mesh.vertices, ori_mesh.faces, points_all, face_norm_all, vert_norm_all


def find_normal(points, index, mesh):
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    all_face_normals = mesh.face_normals
    all_vert_normals = trimesh.geometry.mean_vertex_normals(len(mesh.vertices), mesh.faces, mesh.face_normals, sparse=None)
    #
    # all_vert_normals = mesh.vertex_normals
    print(np.sum(all_vert_normals))
    print(all_vert_normals.shape)
    vertices = mesh.vertices
    faces = mesh.faces
    print("all_face_normals{}, all_vert_normals{}, vertices{}, faces{}, points{}".format(all_face_normals.shape, all_vert_normals.shape, vertices.shape, faces.shape, points.shape))

    face_norms = all_face_normals[index]
    print("face_norms.shape:", face_norms.shape)
    verts_ind = faces[index].reshape(-1) # k X 3 -> 3K
    print("verts_ind.shape:", verts_ind.shape)
    verts_xyz = vertices[verts_ind].reshape(-1,3,3) # 3K * 3
    verts_norm = all_vert_normals[verts_ind].reshape(-1,3,3) # K 3 * 3
    R = rectify_matrix(face_norms)
    allcord = np.concatenate([np.transpose(verts_xyz,(0,2,1)), np.expand_dims(points,axis=2)], axis = 2) # K * 3 * 4
    rotcord = np.matmul(R, allcord)
    verts_inter_norm = baracenter_interp(rotcord[:,0,:3], rotcord[:,1,:3], rotcord[:,0,3], rotcord[:,1,3],verts_norm)
    return face_norms, verts_inter_norm


def interp(points, face_norms, verts_xyz, vert_norms):
    print("interp: points, face_norms, verts_xyz, vert_norms",points.shape, face_norms.shape, verts_xyz.shape, vert_norms.shape)
    R = norm_z_matrix(face_norms)
    allcord = np.concatenate([np.transpose(verts_xyz,(0,2,1)), np.expand_dims(points,axis=2)], axis = 2) # K * 3 * 4
    rotcord = np.matmul(R, allcord)
    verts_inter_norm = baracenter_interp(rotcord[:,0,:3], rotcord[:,1,:3], rotcord[:,0,3], rotcord[:,1,3],vert_norms)
    return verts_inter_norm


def baracenter_interp(X, Y, Px, Py, feats):
    Yv2_v3 = Y[:,1]-Y[:,2]
    Yv1_v3 = Y[:,0]-Y[:,2]
    Px_v3 = Px - X[:,2]
    Xv3_v2 = X[:,2]-X[:, 1]
    Xv1_v3 = X[:,0]-X[:, 2]
    Py_v3 = Py - Y[:,2]
    W1 = (Yv2_v3 * Px_v3 + Xv3_v2 * Py_v3) / (Yv2_v3 * Xv1_v3 + Xv3_v2 * Yv1_v3)
    W2 = (-Yv1_v3 * Px_v3 + Xv1_v3 * Py_v3) / (Yv2_v3 * Xv1_v3 + Xv3_v2 * Yv1_v3)
    W3 = 1 - W1 - W2
    interfeat = W1[:,np.newaxis] * feats[:,0,:] + W2[:,np.newaxis] * feats[:,1,:] + W3[:,np.newaxis] * feats[:,2,:]
    return interfeat


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--thread_num', type=int, default='1', help='how many objs are creating at the same time')
    # parser.add_argument('--shuffle', action='store_true')
    # parser.add_argument('--category', type=str, default="all",
    #                     help='Which single class to generate on [default: all, can be chair or plane, etc.]')
    # FLAGS = parser.parse_args()
    #
    # # nohup python -u gpu_create_ivt.py &> create_sdf.log &
    #
    # #  full set
    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    # if FLAGS.category != "all":
    #     cats = {
    #         FLAGS.category:cats[FLAGS.category]
    #     }
    #
    # create_ivt(32768*2, 0.01, cats, raw_dirs,
    #            lst_dir, uni_ratio=0.5, normalize=True, version=1, skip_all_exist=True)
    #
    norm_mesh_sub_dir = "/ssd1/datasets/ShapeNet/ShapeNetCore_v1_norm/03001627/17e916fc863540ee3def89b32cef8e45"
    verts, faces, surfpoints, facenorm, vertnorm = get_normal(norm_mesh_sub_dir)
    print(np.linalg.norm(facenorm,axis=1))

    save_norm(surfpoints, facenorm, os.path.join("./", "surf_face_norm.ply"))
    save_norm(surfpoints, vertnorm, os.path.join("./", "surf_vert_norm.ply"))
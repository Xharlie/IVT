import os
import numpy as np
from numpy import dot
from math import sqrt
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycuda.driver as drv



def pnts_tries_ivts(pnts, tries, topk_tries=None, gpu=0):
    # print("gpu",gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    if gpu < 0:
        import pycuda.autoinit
    else:
        drv.init()
        dev1 = drv.Device(gpu)
        ctx1 = dev1.make_context()

    mod = SourceModule("""
    #define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

    __device__ void closesPointOnTriangle(float *triangle, float *point, float ivt[5])
    {
            
        float edge0x = triangle[3] - triangle[0];
        float edge0y = triangle[4] - triangle[1];
        float edge0z = triangle[5] - triangle[2];

        float edge1x = triangle[6] - triangle[0];
        float edge1y = triangle[7] - triangle[1];
        float edge1z = triangle[8] - triangle[2];

        float v0x = triangle[0] - point[0];
        float v0y = triangle[1] - point[1];
        float v0z = triangle[2] - point[2];

        float a = edge0x*edge0x + edge0y*edge0y + edge0z*edge0z;
        float b = edge0x*edge1x + edge0y*edge1y + edge0z*edge1z;
        float c = edge1x*edge1x + edge1y*edge1y + edge1z*edge1z;
        float d = edge0x*v0x + edge0y*v0y + edge0z*v0z;
        float e = edge1x*v0x + edge1y*v0y + edge1z*v0z;

        float det = a*c - b*b;
        float s = b*e - c*d;
        float t = b*d - a*e;
        ivt[4] = 1.0;
        if ( s + t < det )
        {
            if ( s < 0.f )
            {
                if ( t < 0.f )
                {
                    if ( d < 0.f )
                    {   
                        t = 0.f;
                        if (-d >= a){
                            s = 1.f;
                        }else{
                            s = -d / a;
                        }
                    }else{
                        s = 0.f;
                        if (e >= 0.f){
                            t = 0.f;
                        }else{
                            if (-e >= c){
                                t = 1.f;
                            }else{
                                t = -e / c;
                            }
                        }
                    }
                }else{
                    s = 0.f;
                    if (e >= 0){
                        t = 0;
                    }else{
                        if (-e >= c){
                            t = 1.f;
                        }else{
                            t = -e / c;
                        }
                    }
                }
            }
            else if ( t < 0.f )
            {
                t = 0.f;
                if (d >= 0){
                    s = 0.f;
                }else{
                    if (-d >= a){
                        s = 1.f;
                    }else{
                        s = -d / a;
                    }
                }
            }
            else
            {
                float invDet = 1.f / det;
                s = s * invDet;
                t = t * invDet;
                ivt[4] = 0.0;
            }
        }
        else
        {
            if ( s < 0.f )
            {
                float tmp0 = b+d;
                float tmp1 = c+e;
                if ( tmp1 > tmp0 )
                {
                    float numer = tmp1 - tmp0;
                    float denom = a-2.0*b+c;
                    if (numer >= denom){
                        s = 1.f;
                        t = 0.f;
                    }else{
                        s = numer / denom;
                        t = 1.f - s;
                    }
                }else{
                    s = 0.f;
                    if (tmp1 <= 0.f){
                        t = 1.f;
                    }else{
                        if (e >= 0.f){
                            t = 0.f;
                        }else{
                            t = -e / c;
                        }
                    }
                }
            }
            else if ( t < 0.f )
            {
                float tmp0 = b + e;
                float tmp1 = a + d;
                if ( tmp1 > tmp0 )
                {
                    float numer = tmp1 - tmp0;
                    float denom = a-2.0*b+c;
                    if (numer >= denom){
                        t = 1.f;
                        s = 0.f;
                    }else{
                        t = numer / denom;
                        s = 1.f - t;
                    }
                }else{
                    t = 0.f;
                    if (tmp1 <= 0.f){
                            s = 1.f;
                    }else{
                        if (d >= 0.f){
                            s = 0.f;
                        }else{
                            s = -d / a;
                        }
                    }
                }
            }
            else
            {
                float numer = c+e-b-d;
                if (numer <= 0){
                    s = 0.f;
                    t = 1.f;
                }else{
                    float denom = a - 2.0 * b + c;
                    if (numer >= denom){
                        s = 1.f;
                        t = 0.f;
                    }else{
                        s = numer / denom;
                        t = 1.f - s;
                    }
                }
            }
        }

        ivt[0] = triangle[0] + s * edge0x + t * edge1x - point[0];
        ivt[1] = triangle[1] + s * edge0y + t * edge1y - point[1];
        ivt[2] = triangle[2] + s * edge0z + t * edge1z - point[2];
        ivt[3] = sqrt(ivt[0]*ivt[0] + ivt[1]*ivt[1] + ivt[2]*ivt[2]);
    }

    __global__ void p2t(float *ivts, float *dist, float *on_edge, float *pnts, float *tries, int pnt_num, int tries_num)
    {
        long i = blockIdx.x * blockDim.x + threadIdx.x;
        int p_id = i / tries_num;
        if (p_id < pnt_num) {
            int t_id = i - p_id * tries_num;
            float triangle[9] = {tries[t_id*9], tries[t_id*9+1], tries[t_id*9+2],tries[t_id*9+3], tries[t_id*9+4], tries[t_id*9+5],tries[t_id*9+6], tries[t_id*9+7], tries[t_id*9+8]};
            float point[3] = {pnts[p_id*3], pnts[p_id*3+1], pnts[p_id*3+2]};
            float ivt[5];
            closesPointOnTriangle(triangle, point, ivt);
            ivts[i*3] = ivt[0];
            ivts[i*3+1] = ivt[1];
            ivts[i*3+2] = ivt[2];
            dist[i] = ivt[3];
            on_edge[i] = ivt[4];
        }
    }
    
    __global__ void p2ttop(float *ivts, float *dist, float *on_edge, float *pnts, float *tries, int pnt_num, int top_num)
    {
        int t_id = blockIdx.x * blockDim.x + threadIdx.x;
        int p_id = t_id / top_num;
        if (p_id < pnt_num) {
            float triangle[9] = {tries[t_id*9], tries[t_id*9+1], tries[t_id*9+2],tries[t_id*9+3], tries[t_id*9+4], tries[t_id*9+5],tries[t_id*9+6], tries[t_id*9+7], tries[t_id*9+8]};
            float point[3] = {pnts[p_id*3], pnts[p_id*3+1], pnts[p_id*3+2]};
            float ivt[5];
            closesPointOnTriangle(triangle, point, ivt);
            ivts[t_id*3] = ivt[0];
            ivts[t_id*3+1] = ivt[1];
            ivts[t_id*3+2] = ivt[2];
            dist[t_id] = ivt[3];
            on_edge[t_id] = ivt[4];
        }
    }
    """)

    kMaxThreadsPerBlock = 1024
    pnt_num = pnts.shape[0]

    if topk_tries is not None:
        top_num = topk_tries.shape[1]
        ivt = np.zeros((pnt_num, top_num, 3)).astype(np.float32)
        dist = 10.0 * np.ones((pnt_num, top_num)).astype(np.float32)
        on_edge = np.zeros((pnt_num, top_num)).astype(np.float32)
        gridSize = int((pnt_num * top_num + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        pnts_tries_ivt = mod.get_function("p2ttop")
        pnts_tries_ivt(drv.Out(ivt), drv.Out(dist), drv.Out(on_edge), drv.In(np.float32(pnts)), drv.In(np.float32(topk_tries)),np.int32(pnt_num), np.int32(top_num), block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
    else:
        print("start to cal ivf", tries.shape)
        tries_num = tries.shape[0]
        ivt = np.zeros((pnt_num, tries_num, 3)).astype(np.float32)
        dist = 10.0 * np.ones((pnt_num, tries_num)).astype(np.float32)
        on_edge = np.zeros((pnt_num, tries_num)).astype(np.float32)
        gridSize = int((pnt_num * tries_num + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
        pnts_tries_ivt = mod.get_function("p2t")
        pnts_tries_ivt(drv.Out(ivt), drv.Out(dist), drv.Out(on_edge), drv.In(np.float32(pnts)), drv.In(np.float32(tries)), np.int32(pnt_num), np.int32(tries_num), block=(kMaxThreadsPerBlock,1,1), grid=(gridSize,1))
    # print("ivt[0,0,:]", ivt[0,0,:])
    if gpu >= 0: 
        ctx1.pop()
    # print("ivt", ivt[0,0,:],ivt[0,1,:])
    # print(pnts[0],pnts[1],tries[0],tries[1])
    # dist0, PP0 = pointTriangleDistance(tries[0], pnts[0])
    # dist1, PP1 = pointTriangleDistance(tries[1], pnts[0])
    # distt2, PP2 = pointTriangleDistance(tries[0], pnts[1])
    # dist1, PP3 = pointTriangleDistance(tries[1], pnts[1])
    # print("PPO", PP0)
    # print("PP1", PP1)
    # print("PP2", PP2)
    # print("PP3", PP3)
    return ivt, dist, on_edge

def cal_topkind(pnts, avg_points, gpu=0):
    if gpu < 0:
        import pycuda.autoinit
    else:
        drv.init()
        dev1 = drv.Device(gpu)
        ctx1 = dev1.make_context()

    mod = SourceModule("""
    __global__ void topk(int *topk_ind, float *pnts, float *avg_points, int pnt_num, int tries_num)
    {
        int p_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (p_id < pnt_num) {
            float topv[10]={10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0}; 
            float topi[10]={INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX,INT_MAX}; 
            float x_dist, y_dist, z_dist, dist, exchangedist;
            float px = pnts[p_id*3];
            float py = pnts[p_id*3+1];
            float pz = pnts[p_id*3+2];
            int distind, exchangeind; 
            for (int t_id=0; t_id<tries_num; t_id++){
                x_dist = px - avg_points[t_id*3];
                y_dist = py - avg_points[t_id*3+1];
                z_dist = pz - avg_points[t_id*3+2];
                dist = sqrt(x_dist*x_dist + y_dist*y_dist + z_dist*z_dist);
                distind = t_id;
                for (int ind=0; ind<10; ind++){
                    if (topv[ind] > dist){
                        exchangedist = topv[ind];
                        topv[ind] = dist;
                        dist = exchangedist;
                        exchangeind = topi[ind];
                        topi[ind] = distind;
                        distind = exchangeind;
                    }
                    if (t_id==(tries_num-1)){
                        topk_ind[p_id*10+ind] = topi[ind];
                    }
                }
            }
        }
    }
    """)
    kMaxThreadsPerBlock = 1024
    pnt_num = pnts.shape[0]
    tries_num = avg_points.shape[0]
    print(avg_points.shape)
    gridSize = int((pnt_num + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock)
    topk_ind = np.zeros((pnt_num, 10)).astype(np.int32)
    pnts_tries_ivt = mod.get_function("topk")
    print(np.int32(tries_num))
    pnts_tries_ivt(drv.Out(topk_ind), drv.In(np.float32(pnts)), drv.In(np.float32(avg_points)), np.int32(pnt_num),
                   np.int32(tries_num), block=(kMaxThreadsPerBlock, 1, 1), grid=(gridSize, 1))
    if gpu >= 0:
        ctx1.pop()
    return topk_ind

def closet(ivt, dist, on_edge):
    # index = (np.arange(ivt.shape[0]) * dist.shape[1] + np.argmin(dist, axis=1))[:,np.newaxis]
    minind = np.argmin(dist, axis=1)
    edge_index = np.arange(on_edge.shape[0]) * on_edge.shape[1] + minind
    ivt_index0 = 3*(np.arange(ivt.shape[0]) * dist.shape[1] + minind)[:,np.newaxis]
    ivt_index1 = ivt_index0 + 1
    ivt_index2 = ivt_index1 + 1
    ivt_indices = np.concatenate([ivt_index0, ivt_index1, ivt_index2],axis=1)
    # print("np.argmin(dist, axis=1)",np.argmin(dist, axis=1))
    # print("index",index)
    # dist_closest = np.take(dist, index)
    ivt_closest = np.take(ivt, ivt_indices)
    on_edge_closest = np.take(on_edge, edge_index)
    # print(index.shape, dist_closest.shape, ivt_closest.shape)
    # print(dist_closest, dist)
    # print(ivt_closest, ivt)
    return ivt_closest, minind, on_edge_closest



def pointTriangleDistance(TRI, P):
    # function [dist,PP0] = pointTriangleDistance(TRI,P)
    # calculate distance between a point and a triangle in 3D
    # SYNTAX
    #   dist = pointTriangleDistance(TRI,P)
    #   [dist,PP0] = pointTriangleDistance(TRI,P)
    #
    # DESCRIPTION
    #   Calculate the distance of a given point P from a triangle TRI.
    #   Point P is a row vector of the form 1x3. The triangle is a matrix
    #   formed by three rows of points TRI = [P1;P2;P3] each of size 1x3.
    #   dist = pointTriangleDistance(TRI,P) returns the distance of the point P
    #   to the triangle TRI.
    #   [dist,PP0] = pointTriangleDistance(TRI,P) additionally returns the
    #   closest point PP0 to P on the triangle TRI.
    #
    # Author: Gwolyn Fischer
    # Release: 1.0
    # Release date: 09/02/02
    # Release: 1.1 Fixed Bug because of normalization
    # Release: 1.2 Fixed Bug because of typo in region 5 20101013
    # Release: 1.3 Fixed Bug because of typo in region 2 20101014

    # Possible extention could be a version tailored not to return the distance
    # and additionally the closest point, but instead return only the closest
    # point. Could lead to a small speed gain.

    # Example:
    # %% The Problem
    # P0 = [0.5 -0.3 0.5]
    #
    # P1 = [0 -1 0]
    # P2 = [1  0 0]
    # P3 = [0  0 0]
    #
    # vertices = [P1; P2; P3]
    # faces = [1 2 3]
    #
    # %% The Engine
    # [dist,PP0] = pointTriangleDistance([P1;P2;P3],P0)
    #
    # %% Visualization
    # [x,y,z] = sphere(20)
    # x = dist*x+P0(1)
    # y = dist*y+P0(2)
    # z = dist*z+P0(3)
    #
    # figure
    # hold all
    # patch('Vertices',vertices,'Faces',faces,'FaceColor','r','FaceAlpha',0.8)
    # plot3(P0(1),P0(2),P0(3),'b*')
    # plot3(PP0(1),PP0(2),PP0(3),'*g')
    # surf(x,y,z,'FaceColor','b','FaceAlpha',0.3)
    # view(3)

    # The algorithm is based on
    # "David Eberly, 'Distance Between Point and Triangle in 3D',
    # Geometric Tools, LLC, (1999)"
    # http:\\www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
    #
    #        ^t
    #  \     |
    #   \reg2|
    #    \   |
    #     \  |
    #      \ |
    #       \|
    #        *P2
    #        |\
    #        | \
    #  reg3  |  \ reg1
    #        |   \
    #        |reg0\
    #        |     \
    #        |      \ P1
    # -------*-------*------->s
    #        |P0      \
    #  reg4  | reg5    \ reg6
    # rewrite triangle in normal form
    B = TRI[0, :]
    E0 = TRI[1, :] - B
    # E0 = E0/sqrt(sum(E0.^2)); %normalize vector
    E1 = TRI[2, :] - B
    # E1 = E1/sqrt(sum(E1.^2)); %normalize vector
    D = B - P
    a = dot(E0, E0)
    b = dot(E0, E1)
    c = dot(E1, E1)
    d = dot(E0, D)
    e = dot(E1, D)
    f = dot(D, D)

    #print "{0} {1} {2} ".format(B,E1,E0)
    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    # Terible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f;  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f;  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    if sqrdistance < 0:
        sqrdistance = 0

    dist = sqrt(sqrdistance)

    PP0 = B + s * E0 + t * E1 - P

    return dist, PP0




# ivt = pnts_tries_ivts([],[])
# ivt_py = np.zeros_like(ivt)
# dist_py = np.zeros_like(dist)
# for i in range(pnt_num):
#     pnt = pnts[i]
#     for j in range(tries_num):
#         tri = tries[j]
#         d0, p0 = pointTriangleDistance(tri, pnt)
#         ivt_py[i,j] = p0
#         dist_py[i,j] = d0
# # print(ivt-ivt_py)
# # print(ivt-ivt_py)
# diffdist= dist-dist_py
# diff = np.argmax(abs(diffdist).flatten)
# wi, wj = diff // j, diff % j
# if abs(diffdist[wi,wj])>0.01:
#     print(diffdist)
#     print("diff,indx", diffdist[wi,wj],diff)
#     print("wi, wj", wi,wj)
#     print("pnts[wi], tries[wj]", pnts[wi], tries[wj])
#     print("ivt[wi,wj], ivt_py[wi,wj]", ivt[wi,wj], ivt_py[wi,wj])
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.set_aspect('equal')
#     ax.plot([tries[wj,0,0], tries[wj,1,0]],
#             [tries[wj,0,1],tries[wj,1,1]],
#             [tries[wj,0,2],tries[wj,1,2]])
#     ax.plot([tries[wj,1,0], tries[wj,2,0]],
#             [tries[wj,1,1],tries[wj,2,1]],
#             [tries[wj,1,2],tries[wj,2,2]])
#     ax.plot([tries[wj,2,0], tries[wj,0,0]],
#             [tries[wj,2,1],tries[wj,0,1]],
#             [tries[wj,2,2],tries[wj,0,2]])
#     ax.plot([pnts[wi,0], pnts[wi,0] + ivt[wi,wj,0]],
#             [pnts[wi,1], pnts[wi,1] + ivt[wi,wj,1]],
#             [pnts[wi,2], pnts[wi,2] + ivt[wi,wj,2]],'g')
#     ax.plot([pnts[wi,0], pnts[wi,0] + ivt_py[wi,wj,0]],
#             [pnts[wi,1], pnts[wi,1] + ivt_py[wi,wj,1]],
#             [pnts[wi,2], pnts[wi,2] + ivt_py[wi,wj,2]],'r')
#     X = np.array([tries[wj,0,0], tries[wj,1,0], tries[wj,2,0],pnts[wi,0], pnts[wi,0] + ivt[wi,wj,0],pnts[wi,0] + ivt_py[wi,wj,0]])
#     Y = np.array([tries[wj,0,1], tries[wj,1,1], tries[wj,2,1],pnts[wi,1], pnts[wi,1] + ivt[wi,wj,1],pnts[wi,1] + ivt_py[wi,wj,1]])
#     Z = np.array([tries[wj,0,2], tries[wj,1,2], tries[wj,2,2],pnts[wi,2], pnts[wi,2] + ivt[wi,wj,2],pnts[wi,2] + ivt_py[wi,wj,2]])
#     max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
#     mid_x = (X.max()+X.min()) * 0.5
#     mid_y = (Y.max()+Y.min()) * 0.5
#     mid_z = (Z.max()+Z.min()) * 0.5
#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)
#     plt.show()
#     Axes3D.plot()


# i,j = 2000, 4000
# pnts = np.random.randn(i, 3).astype(np.float32)
# tries = np.random.randn(j, 3, 3).astype(np.float32)
# pnts = np.array([[ 0.0388784,  -0.43049761,  0.93390526], [ 0.65514363,  0.33294779, -0.46959095]])
# tries = np.array([[[ 0.59400288, -0.48788681,  0.40524953],
#  [ 0.53634743, -0.4928393,   0.40898072],
#  [ 0.59449531, -0.4878803,   0.41286809]], [[ 0.59449531, -0.4878803,   0.41286809],
#  [ 0.53634743, -0.4928393,   0.40898072],
#  [ 0.59300718, -0.48804083,  0.42035432]]])
# ivt, dist = pnts_tries_ivts(pnts, tries)
# closet(ivt, dist)
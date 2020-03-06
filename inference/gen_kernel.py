import numpy as np
import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../preprocessing'))
import create_ivt as ct
import pymesh
import trimesh

def get_unigrid(ivt_res):
    grids = int(1 / ivt_res) 
    x_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    y_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    z_ = np.linspace(-1, 1, num=grids).astype(np.float32)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    return np.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)

def get_normalized_mesh(model_file):
	total = 16384 * 5
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
	ori_mesh = pymesh.load_mesh(model_file)
	index = ori_mesh.faces.reshape(-1)
	tries = ori_mesh.vertices[index].reshape([-1,3,3])
	return points_all, tries


def unisample_pnts(res, model_file, threshold=0.1, stdratio=10):
	gt_pnts, tries = get_normalized_mesh(model_file)
	unigrid = get_unigrid(res)
	uni_ivts = ct.calculate_ivt(unigrid, tries)
	uni_dist = np.linalg.norm(uni_ivts, axis=1)
	ind = uni_dist <= threshold
	uni_place, std = cal_std_loc(unigrid[ind], uni_ivts[ind], stdratio)
	return uni_place, std, np.full(std.shape[0], 1/std.shape[0]), gt_pnts, tries


def cal_std_loc(pnts, ivts, stdratio):
	loc = pnts + ivts
	dist = np.linalg.norm(ivts, axis=1)
	std = dist / stdratio
	return loc, np.tile(np.expand_dims(std,axis=1),(1,3))

def sample_from_GMM(mean_loc, std, weights, num):
	inds = np.random.choice(weights.shape[0], num, p=weights)
	sampled_pnts = np.zeros((num, 3))
	for i in range(num):
		ind = inds[i]
		loc = np.random.normal(mean_loc[ind],std[ind])
		sampled_pnts[i,:] = loc
	return sampled_pnts

def nearsample_pnts(locs, tries, stdratio=10):
	surface_ivts = ct.calculate_ivt(locs, tries)
	surface_place, std = cal_std_loc(locs, surface_ivts, stdratio)
	return surface_place, std, np.full(std.shape[0], 1/std.shape[0])

if __name__ == "__main__":

    # nohup python -u create_point_sdf_grid.py &> create_sdf.log &

    #  full set
    # lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()
    # if FLAGS.category != "all":
    #     cats = {
    #         FLAGS.category:cats[FLAGS.category]
    #     }

    # create_ivt(32768, 0.01, cats, raw_dirs,
    #            lst_dir, uni_ratio=0.2, normalize=True, version=1, skip_all_exist=True)
    rounds = 5
    nums = 10000
    num_ratio = 2

    mean_loc, std, weights, gt_pc, tries = unisample_pnts(0.02, "/ssd1/datasets/ShapeNet/ShapeNetCore_v1_norm/04530566/5d48d75153eb221b476c772fd813166d/pc_norm.obj", threshold=0.1, stdratio=2)
    pc = sample_from_GMM(mean_loc, std, weights, nums)
    np.savetxt("gt_pc.txt", gt_pc, delimiter=';')
    np.savetxt("uni_pc.txt", pc, delimiter=';')
    for i in range(rounds):
    	loc, std, weights = nearsample_pnts(pc, tries, stdratio=2)
    	pc = sample_from_GMM(loc, std, weights, nums*num_ratio**i)
    	print("pc.shape", pc.shape, "loc.shape", loc.shape, "std.shape", std.shape, "weights.shape", weights.shape)
    	np.savetxt("surf_pc{}.txt".format(i), pc, delimiter=';')

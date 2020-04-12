import h5py
import numpy as np
import create_file_lst
import os,string
import sys

def move(cats,norm_mesh_dir,pnt_dir):
	old_pnt_dir = "/ssd1/datasets/ShapeNet/pnt_old"
	if not os.path.exists(pnt_dir): os.makedirs(pnt_dir)
	for catnm in cats.keys():
		cat_id = cats[catnm]
		cat_pnt_dir = os.path.join(pnt_dir, cat_id)
		cat_norm_mesh_dir = os.path.join(norm_mesh_dir, cat_id)
		cat_old_pnt_dir = os.path.join(old_pnt_dir, cat_id)
		if not os.path.exists(cat_pnt_dir): os.makedirs(cat_pnt_dir)
		with open(lst_dir + "/" + str(cat_id) + "_test.lst", "r") as f:
			list_obj = f.readlines()
		with open(lst_dir + "/" + str(cat_id) + "_train.lst", "r") as f:
			list_obj += f.readlines()
		for obj in list_obj:
			obj = obj.strip()
			obj_pnt_dir = os.path.join(cat_pnt_dir,obj)
			obj_norm_mesh_dir = os.path.join(cat_norm_mesh_dir,obj)
			obj_old_pnt_dir = os.path.join(cat_old_pnt_dir,obj)
			norm_file = obj_norm_mesh_dir+"/pnt_163840.h5"
			obj_old_pnt_file = obj_old_pnt_dir+"/pnt_163840.h5"
			if not os.path.exists(obj_pnt_dir): os.makedirs(obj_pnt_dir)
			if not os.path.exists(obj_pnt_dir):
				if os.path.exists(norm_file):
					command_str = "mv " + norm_file + " " + obj_pnt_dir + "/pnt_163840.h5"
					print("command:", command_str)
					os.system(command_str)
				elif os.path.exists(obj_old_pnt_file):
					command_str = "mv " + obj_old_pnt_file + " " + obj_pnt_dir + "/pnt_163840.h5"
					print("command:", command_str)
					os.system(command_str)


lst_dir, cats, all_cats, raw_dirs = create_file_lst.get_all_info()

norm_mesh_dir=raw_dirs["norm_mesh_dir"]
pnt_dir=raw_dirs["pnt_dir"]
move(cats,norm_mesh_dir,pnt_dir)



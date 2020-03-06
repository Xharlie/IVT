import h5py
import numpy as np

h5file = "/ssd1/datasets/ShapeNet/IVT_v1/04530566/eba55caf770565989c063286c702ba92/ivt_sample.h5"

def generate_surface(point_ds, ivt_ds, h5_file):
	    f = h5py.File(h5_file, 'r')
	    points = f[point_ds][:]
	    ivt = f[ivt_ds][:]
	    rec_pnts = points + ivt
	    f.close()
	    return points, ivt, rec_pnts

uni_pnts, uni_ivts, uni_rec_pnts = generate_surface("uni_pnts","uni_ivts", h5file)

surf_pnts, surf_ivts, surf_rec_pnts = generate_surface("surf_pnts","surf_ivts", h5file)

combine_pnts = np.concatenate([uni_pnts, surf_pnts], axis=0)
rec_combine_pnts = np.concatenate([uni_rec_pnts, surf_rec_pnts], axis=0)

np.savetxt("uni_pnts.txt", uni_pnts, delimiter=';')
np.savetxt("surf_pnts.txt", surf_pnts, delimiter=';')
np.savetxt("uni_rec_pnts.txt", uni_rec_pnts, delimiter=';')
np.savetxt("surf_rec_pnts.txt", surf_rec_pnts, delimiter=';')
np.savetxt("combine_pnts.txt", combine_pnts, delimiter=';')
np.savetxt("rec_combine_pnts.txt", rec_combine_pnts, delimiter=';')
ind = np.random.randint(uni_pnts.shape[0], size=10)
single_pnts = uni_pnts[ind]
single_rec_pnts = uni_rec_pnts[ind]
single_surf_pnts = surf_pnts[ind]
single_surf_rec_pnts = surf_rec_pnts[ind]
np.savetxt("single_pnts.txt", single_pnts, delimiter=';')
np.savetxt("single_rec_pnts.txt", single_rec_pnts, delimiter=';')
np.savetxt("single_surf_pnts.txt", single_surf_pnts, delimiter=';')
np.savetxt("single_surf_rec_pnts.txt", single_surf_rec_pnts, delimiter=';')

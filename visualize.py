import torch
import numpy as np
from src.utils.plot_utils import visual_mps, visual_mask
from src.utils.data_utils import kspace_to_sensmaps_mvue
from sigpy.mri.app import EspiritCalib

highres_file_path = "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1_coilmap_highres.npy"
smap_high = np.load(highres_file_path)
visual_mps(smap_high, 'highres_coilmap.png')

lowres_file_path = "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1_coilmap_lowres.npy"
smap_low = np.load(lowres_file_path)
visual_mps(smap_low, 'lowres_coilmap.png')

full_kspace_path = "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1.npy"
kspace = np.load(full_kspace_path)
_,sens_map = kspace_to_sensmaps_mvue(kspace)
visual_mps(sens_map, 'bart_coilmap.png')

app = EspiritCalib(kspace)
sens_map_sigpy = app.run()
visual_mps(sens_map_sigpy, 'sigpy_coilmap.png')

mask_gaus_05_3_path = "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/sampling/gaussian_0.5_3.pt"
mask_gaus_05_5_path = "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/sampling/gaussian_0.5_5.pt"
mask_gaus_05_3_tensor = torch.load(mask_gaus_05_3_path, map_location='cpu')
mask_gaus_05_5_tensor = torch.load(mask_gaus_05_5_path, map_location='cpu')
mask_gaus_05_3_np = mask_gaus_05_3_tensor.cpu().numpy()
mask_gaus_05_5_np = mask_gaus_05_5_tensor.cpu().numpy()
visual_mask(mask_gaus_05_3_np, "mask_gaus_05_3.png")
visual_mask(mask_gaus_05_5_np, "mask_gaus_05_5.png")
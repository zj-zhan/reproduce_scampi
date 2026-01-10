import os
import sys
from pathlib import Path
import torch
import numpy as np
from src.ucnnreco import CartesianScampi, NonCartesianScampi
from src.utils.params import RecoParams
from src.utils.util_eval import mae,mse,nmse,psnr,ssim

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    config_path = "/data0/zijian/project/github/reproduce_scampi/src/config/CScampi.json"
    cartesian_params = RecoParams()
    cartesian_params.from_json(config_path) 
    print("Loaded parameters:")
    print(cartesian_params)

    reco = CartesianScampi(recopars=cartesian_params, device=device)
    reco.set_data_path(
        full_kspace_path=Path('data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1'),
        mask_path=Path('data/cartesian/brain_209_6001331/sampling/gaussian_0.5_3'),
        coilmap_path=Path('data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1_coilmap_lowres')
    )

    # Load the data
    reco.prep_data()

    #Preparing model
    reco.prep_model()
    gt = reco.get_gt()

    #Starting reconstruction
    res = reco()

    print("res.shape:",res.shape)
    print("gt.shape:",gt.shape)
    gt_np = gt.detach().cpu().abs().numpy().squeeze()
    res_np = res.detach().cpu().abs().numpy().squeeze()
    nmse_val = nmse(gt_np, res_np)
    psnr_val = psnr(gt_np, res_np, np.max(gt_np))
    ssim_val = ssim(gt_np, res_np, np.max(gt_np))
    print("nmse:",nmse_val)
    print("psnr:",psnr_val)
    print("ssim:",ssim_val)
if __name__ == "__main__":
    main()
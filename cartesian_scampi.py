import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
from pathlib import Path
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.mridataset import MRIDataset
from src.ucnnreco import CartesianScampi, NonCartesianScampi
from src.utils.params import RecoParams
from src.utils.util_eval import mae,mse,nmse,psnr,ssim

def load_pt_mask(mask_path, device):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    mask_t = torch.load(mask_path, map_location='cpu')
    if isinstance(mask_t, np.ndarray):
        mask_t = torch.from_numpy(mask_t)
    mask_t = mask_t.squeeze()
    if mask_t.ndim > 2:
        if mask_t.shape[0] < mask_t.shape[-1]: 
            mask_t = mask_t[0, ...] 
        else:
            mask_t = mask_t[..., 0]
    return mask_t.float().to(device)

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config_path = 'src/config/CScampi.json' 
    cartesian_params = RecoParams()
    cartesian_params.from_json(config_path)
    print(cartesian_params)

    mask = load_pt_mask(args.mask_path, device)

    dataset = MRIDataset(
        root=args.data_root,
        target_size=396, 
        which_data="brain",
        data_norm_type="volume_max",
        device=device
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    metric_log = {
        "nmse": [],
        "psnr": [],
        "ssim": [],
    }

    pbar = tqdm(dataloader, total=len(dataloader), desc="Reconstructing")

    for i, batch in enumerate(pbar):
        kspace_t = batch['kspace'].squeeze(0).to(device)
        mps_t    = batch['mps'].squeeze(0).to(device)
        fname    = batch['fname'][0]
        data_dict = {
                'full_kspace': kspace_t,
                'mask': mask,
                'coilmap': mps_t,
                'us_kspace': None
            }

        scampi = CartesianScampi(cartesian_params, data_dict, device)
        scampi.prep_data()
        scampi.prep_model()
        res = scampi.forward()
        gt_mag = torch.abs(scampi.gt).squeeze().cpu().numpy()
        res_mag = torch.abs(res).squeeze().cpu().numpy()
        cur_nmse = nmse(gt_mag, res_mag)
        cur_psnr = psnr(gt_mag, res_mag, gt_mag.max())
        cur_ssim = ssim(gt_mag, res_mag, gt_mag.max())

        metric_log["nmse"].append(cur_nmse)
        metric_log["psnr"].append(cur_psnr)
        metric_log["ssim"].append(cur_ssim)

    avg_nmse = np.mean(metric_log["nmse"])
    avg_psnr = np.mean(metric_log["psnr"])
    avg_ssim = np.mean(metric_log["ssim"])
    std_nmse = np.std(metric_log["nmse"])
    std_psnr = np.mean(metric_log["psnr"])
    std_ssim = np.std(metric_log["ssim"])
    
    print(f"Total Processed: {len(metric_log['psnr'])} slices")
    print(f"NMSE: {avg_nmse:.5f} +/- {std_nmse:5f}")
    print(f"PSNR: {avg_psnr:.5f} +/- {std_psnr:5f}")
    print(f"SSIM: {avg_ssim:.5f} +/- {std_ssim:.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default= "/data0/zijian/data/DDS_data/exp-fmbrain")
    parser.add_argument('--mask_path', type=str, default= "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/sampling/gaussian_0.5_3.pt")
    
    args = parser.parse_args()
    main(args)
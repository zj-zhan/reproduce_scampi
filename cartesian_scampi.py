import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import shutil
from pathlib import Path
import torch
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.mridataset import MRIDataset
from src.ucnnreco import CartesianScampi, NonCartesianScampi
from src.utils.params import RecoParams
from src.utils.util_eval import nmse,psnr,ssim,get_mask
from src.utils.util_eval import center_crop,img3_rm_black_border
from src.utils.plot_utils import plot_gt_pred

def main(args):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sampling: Type={args.mask_type}, Acc={args.acc_factor}x, Center={args.center_fraction}")

    config_path = 'src/config/CScampi.json' 
    cartesian_params = RecoParams()
    cartesian_params.from_json(config_path)
    print(cartesian_params)

    #mask = load_pt_mask(args.mask_path, device)
    #mask = mask.T
    #mask_visual = mask.cpu().numpy()
    #visual_mask(mask_visual, "mask.png")

    dataset = MRIDataset(
        root=args.data_root,
        target_size=args.target_size, 
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
        "psnr2d": [],
        "ssim2d": [],
        "psnr3d": [],
        "ssim3d": [],
    }

    pbar = tqdm(dataloader, total=len(dataloader), desc="Reconstructing")

    dummy_img = torch.zeros(1, 1, args.target_size, args.target_size).to(device)

    for i, batch in enumerate(pbar):
        kspace    = batch['kspace'].squeeze(0).to(device)
        rss       = batch['rss'].squeeze(0)
        maxval    = batch['max_val'].squeeze(0).item()
        mvue      = batch['mvue'].squeeze(0)
        shape_raw = batch["shape_raw"]
        mask = get_mask(
            img=dummy_img, 
            size=args.target_size, 
            batch_size=1, 
            type=args.mask_type, 
            acc_factor=args.acc_factor, 
            center_fraction=args.center_fraction,
            fix=False 
        )
        mask = mask.squeeze()
        data_dict = {
                'full_kspace': kspace,
                'mask': mask,
                'coilmap': None,
                'us_kspace': None
            }

        scampi = CartesianScampi(cartesian_params, data_dict, device)
        scampi.prep_data()
        scampi.prep_model()
        mps = scampi.coilmap
        mps = mps.squeeze().cpu().numpy()
        res = scampi()


        #gt = torch.abs(scampi.get_gt()).squeeze().cpu().numpy() #for mvue eval
        #eval on mvue (gt same as DDS)
        gt = np.abs(mvue).squeeze().cpu().numpy() 
        res = torch.abs(res).squeeze().cpu().numpy()
        y = gt
        x = res

        #eval on rss
        #gt = rss.squeeze().cpu().numpy() # for rss eval
        #res = res.squeeze().cpu().numpy()
        #recon_coil_imgs = res * mps
        #rss_recon = np.sqrt(np.sum(np.abs(recon_coil_imgs)**2, axis=0)) #for rss eval
        #y = gt
        #x = rss_recon

        target_shape = shape_raw[0].cpu().numpy()

        y = center_crop(y, target_shape)
        x = center_crop(x, target_shape)
        y, x, _ = img3_rm_black_border(y, x)

        maxval0 = float(np.max(y)) #for 2d eval
        maxval1 = maxval #for 3d eval

        cur_nmse = nmse(y, x)
        cur_psnr2d = psnr(y, x, maxval0)
        cur_ssim2d = ssim(y, x, maxval0)
        cur_psnr3d = psnr(y, x, maxval1)
        cur_ssim3d = ssim(y, x, maxval1)
        metric_log["nmse"].append(cur_nmse)
        metric_log["psnr2d"].append(cur_psnr2d)
        metric_log["ssim2d"].append(cur_ssim2d)
        metric_log["psnr3d"].append(cur_psnr3d)
        metric_log["ssim3d"].append(cur_ssim3d)

        if i < 80:
            plot_gt_pred(
                gt=y,
                pred=x,
                shape_raw=None,
                max_value=maxval1,
                output_dir=args.output_dir,
                name_ids=i,
                escale=10.
            )

    avg_nmse = np.mean(metric_log["nmse"])
    avg_psnr2d = np.mean(metric_log["psnr2d"])
    avg_ssim2d = np.mean(metric_log["ssim2d"])
    avg_psnr3d = np.mean(metric_log["psnr3d"])
    avg_ssim3d = np.mean(metric_log["ssim3d"])
    std_nmse = np.std(metric_log["nmse"])
    std_psnr2d = np.std(metric_log["psnr2d"])
    std_ssim2d = np.std(metric_log["ssim2d"])
    std_psnr3d = np.std(metric_log["psnr3d"])
    std_ssim3d = np.std(metric_log["ssim3d"])
    
    print(f"Total Processed: {len(metric_log['nmse'])} slices")
    print(f"NMSE: {avg_nmse:.5f} +/- {std_nmse:5f}")
    print(f"PSNR2d: {avg_psnr2d:.5f} +/- {std_psnr2d:5f}")
    print(f"SSIM2d: {avg_ssim2d:.5f} +/- {std_ssim2d:.5f}")
    print(f"PSNR3d: {avg_psnr3d:.5f} +/- {std_psnr3d:5f}")
    print(f"SSIM3d: {avg_ssim3d:.5f} +/- {std_ssim3d:.5f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default= "/data0/zijian/data/DDS_data/exp-fmbrain")
    #parser.add_argument('--mask_path', type=str, default= "/data0/zijian/project/github/reproduce_scampi/data/cartesian/brain_209_6001331/sampling/gaussian_0.5_3.pt")
    parser.add_argument('--output_dir', type=str, default='./results/exp2_equi320_mvue_rss')
    parser.add_argument('--mask_type', type=str, default='equispaced1d')
    parser.add_argument('--acc_factor', type=float, default=4.0)
    parser.add_argument('--center_fraction', type=float, default=0.08)
    parser.add_argument('--target_size', type=int, default=320)
    
    args = parser.parse_args()
    main(args)
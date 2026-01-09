import os
import sys
from pathlib import Path
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)
from ucnnreco import CartesianScampi, NonCartesianScampi
from utils.plot_utils import disp
from utils.params import RecoParams
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr(gt: np.ndarray, pred: np.ndarray, maxval: float):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(gt: np.ndarray, pred: np.ndarray, maxval: float):
    """Compute Structural Similarity Index Metric (SSIM)"""
    assert gt.ndim == 3
    Nc, Ny, Nx = gt.shape
    y = 0.0
    for i in range(Nc):
        y = y + structural_similarity(gt[i,:,:], pred[i,:,:], data_range=maxval)
    y = y / Nc
    return y

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    cartesian_params = RecoParams()
    config_path = os.path.join(current_dir, 'src', 'config', 'CScampi.json')
    cartesian_params.from_json(config_path)    
    print("Loaded parameters:")
    print(cartesian_params)

    reco = CartesianScampi(recopars=cartesian_params, device=device)
    reco.set_data_path(
        full_kspace_path=Path('data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1'),
        mask_path=Path('data/cartesian/brain_209_6001331/sampling/gaussian_0.5_3'),
        coilmap_path=Path('data/cartesian/brain_209_6001331/file_brain_AXT2_209_6001331_1_coilmap_highres')
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
    gt_np = gt.detach().cpu().abs().numpy()
    res_np = res.detach().cpu().abs().numpy()
    psnr_val = psnr(gt_np, res_np, np.max(gt_np))
    ssim_val = ssim(gt_np, res_np, np.max(gt_np))
    print("psnr:",psnr_val)
    print("ssim:",ssim_val)
if __name__ == "__main__":
    main()
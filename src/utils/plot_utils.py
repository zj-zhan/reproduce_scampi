import os
import math
import torch
import numpy as np
import matplotlib.cm as cmc
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from src.utils.data_utils import mda_slice
from src.utils.util_eval import nmse, psnr, ssim


def disp(input1, input2=None, dims=(1, 2), scale_error=1.0, log=False):
    # Ensure inputs are numpy arrays
    if torch.is_tensor(input1):
        input1 = input1.cpu().numpy()
    if input2 is not None and torch.is_tensor(input2):
        input2 = input2.cpu().numpy()

    if len(input1.shape) > 2:
        # Select two dimensions for multidimensional arrays
        input1 = mda_slice(input1, dims)
    # Calculate magnitude and phase for input1
    if not log:
        mag1 = np.abs(input1)
    else:
        mag1 = np.log(np.abs(input1) + .002)

    phase1 = np.angle(input1)

    plt.figure(figsize=(10, 10))
    plt.gray()

    if input2 is not None:
        if len(input2.shape) > 2:
            # Select two dimensions for multidimensional arrays
            input2 = mda_slice(input2, dims)

        phase2 = np.angle(input2)

        if not log:
            mag2 = np.abs(input2)
        else:
            mag2 = np.log(np.abs(input2) + .002)

        # Calculate difference
        diff = np.abs(mag1 - mag2)

        # Plotting
        plt.subplot(3, 2, 1)
        plt.imshow(mag1.transpose())
        plt.title('Magnitude of Input1')
        plt.colorbar()

        plt.subplot(3, 2, 2)
        plt.imshow(phase1.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase of Input1')
        plt.colorbar()

        plt.subplot(3, 2, 3)
        plt.imshow(mag2.transpose())
        plt.title('Magnitude of Input2')
        plt.colorbar()

        plt.subplot(3, 2, 4)
        plt.imshow(phase2.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase of Input2')
        plt.colorbar()

        plt.subplot(3, 2, 5)
        plt.imshow(diff.transpose(), vmin=0, vmax=mag1.max() / scale_error)
        plt.title(' Error-Map (scaling : ' + str(scale_error) + ')')
        plt.colorbar()

    else:
        # Plotting
        plt.subplot(2, 1, 1)
        plt.imshow(mag1.transpose())
        plt.title('Magnitude')
        plt.colorbar()

        plt.subplot(2, 1, 2)
        plt.imshow(phase1.transpose(), vmin = -math.pi, vmax = math.pi)
        plt.title('Phase')
        plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_traj(traj, n_spokes=40, spoke_length=256):
    tr = traj.cpu().numpy()

    # plot the first 40 spokes
    kx = tr[0, :n_spokes * spoke_length]
    kx = np.reshape(kx, (n_spokes, spoke_length))
    ky = traj[1, :n_spokes * spoke_length]
    ky = np.reshape(ky, (n_spokes, spoke_length))

    viridis = cmc.get_cmap('viridis', n_spokes)
    colors = viridis(np.linspace(0, 2, n_spokes))
    for i in range(n_spokes):
        plt.scatter(kx[i, :], ky[i, :], s=3, color=colors[i])

    # plt.scatter(kx[:40, :].transpose(), ky[:40, :].transpose())
    plt.axis('equal')
    plt.title('k-space trajectory (first {n_spokes} spokes)')
    plt.show()

def plot_gt_pred(gt, pred, shape_raw, max_value, output_dir, name_ids, escale=10.):
    """
    Plot ground truth image with predicted image, both with 1 channel.
    Image black border is cut if exists in GT. Metrics are saved in filename.

    Parameters:
        gt: np.array [H, W]
        pred: np.array [H, W]
        shape_raw: [Nbs, 2], reference shape for center crop
        max_value: float
        output_dir: str, path to save figures
        escale: float, error display range scaling factor
    """
    os.makedirs(output_dir, exist_ok=True)
    erro = np.abs(gt - pred)

    plt.rcParams.update({'font.size': 14})
    title_list = ["True", "Predict", "Error"]

    with plt.ioff():
        fig, axs = plt.subplot_mosaic([['a'], ['b'], ['c']], layout='constrained',
                                        figsize=(6, 15), dpi=300)
        target_shape = shape_raw
        #gts = center_crop(gt, target_shape)
        #preds = center_crop(pred, target_shape)
        #erros = center_crop(erro, target_shape)
        #gts, preds, erros = img3_rm_black_border(gts, preds, erros)
        maxval = max_value
        vr = [0.0, np.max(gt)]

        for i, (label, ax) in enumerate(axs.items()):
            if i == 0:
                a = ax.imshow(gt, vmin=vr[0], vmax=vr[1], cmap='gray')
            elif i == 1:
                a = ax.imshow(pred, vmin=vr[0], vmax=vr[1], cmap='gray')
            elif i == 2:
                a = ax.imshow(erro, vmin=vr[0], vmax=vr[1]/escale, cmap='jet')
            ax.set_axis_off()
            ax.set_title(title_list[i])
            plt.colorbar(a, location='right')

        nmse_val = nmse(gt, pred)
        psnr_val = psnr(gt, pred, maxval)
        ssim_val = ssim(gt, pred, maxval)
        metric_str = f"nmse_{nmse_val:.5f}_psnr_{psnr_val:.4f}_ssim_{ssim_val:.4f}"
        savename = os.path.join(output_dir, f"data_{name_ids}_{metric_str}.png")
        plt.savefig(savename, bbox_inches='tight')
        plt.close()

    return None

def visual_mps(mps, save_path):
    n_coils, h, w = mps.shape
    mag_stack = np.abs(mps).transpose(1, 0, 2).reshape(h, -1)
    phase_stack = np.angle(mps).transpose(1, 0, 2).reshape(h, -1)

    fig_width = max(10, n_coils * 2)
    fig_height = 5
    
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))

    im0 = axes[0].imshow(mag_stack, cmap='gray', vmin=0)
    axes[0].set_title(f"Sensitive map Magnitude")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase_stack, cmap='gray', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title(f"Sensitive map Phase")
    axes[1].axis('off')
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[-np.pi, 0, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', '0', r'$\pi$'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visual_mask(mask, save_path):

    mask_disp = np.squeeze(mask)
    if mask_disp.ndim > 2:
        mask_disp = mask_disp[0, ...]

    pe_lines, fe_lines = mask_disp.shape
    sampled_lines = int(np.sum(mask_disp[:, 0]))
    total_pe_lines = pe_lines
    acceleration = total_pe_lines / sampled_lines if sampled_lines > 0 else 0
    mask_to_plot = mask_disp.T 
    fig, ax = plt.subplots(figsize=(8, 8)) 
    im = ax.imshow(mask_to_plot, cmap='gray', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"K-space Mask Visualization\nShape: {mask_disp.shape}, R ≈ {acceleration:.2f}x\nSampled Lines: {sampled_lines} / {total_pe_lines}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visual_kspace(kspace, save_path):
    n_coils, h, w = kspace.shape
    kspace = np.log10(np.abs(kspace) + 1e-9)
    stacked_img = kspace.transpose(1, 0, 2).reshape(h, -1)

    fig_width = max(10, n_coils * 2) 
    fig_height = 5
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    im = ax.imshow(stacked_img, cmap='gray')
    ax.set_title("Log(|kspace|)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visual_rss(rss, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(rss, cmap='gray')
    ax.set_title("RSS")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
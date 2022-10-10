from __future__ import print_function, absolute_import, division

import math

import apex
import torch
import torch.nn as nn
import torchgeometry as tgm
import matplotlib.pyplot as plt

from tqdm import tqdm


from opts import get_parse_args
from functions.utils import save_ckpt, AverageMeter, get_pixel_dist
from functions.loss import JointsMSELoss, epe, p_epe

from utils import get_model, get_dataset, get_optimizer, get_lr_scheduler, init, set_A_binary
from data_utils import align_bone_xaxis


def main(args):
    set_A_binary(args.dataset)
    device = torch.device("cuda")
    print('==> Creating DataLoader')
    train_loader, val_loader, k_set, num_joint, percentile = get_dataset(args)
    print("==> Creating PoseNet model...")
    self_sup = get_model(args, num_joint, device)
    print("==> Prepare optimizer...")
    optimizer = get_optimizer(args, self_sup)
    print("==> Creating Scheduler...")
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_loader))
    kpt2d_recon_loss = JointsMSELoss()

    if args.half:
        self_sup, optimizer = apex.amp.initialize(
            self_sup,
            optimizer,
            opt_level=f'O{args.amp_opt_level}'
        )

    ckpt_dir_path = args.work_dir

    #################################################
    # ########## start training here
    #################################################
    start_epoch = 0
    error_best = None

    bar = tqdm(val_loader, desc='Evaluate 2D keypoint predictions', total=len(val_loader))
    epe_error = AverageMeter()
    p_epe_error = AverageMeter()
    for gt, kpt2d_pred, meta in bar:
        kpt2d = gt[:,:,:2].to(device)
        kpt2d_pred = kpt2d_pred.to(device)

        kpt2d_pred = kpt2d_pred[:,:,:2]
        img_size = torch.stack((meta['cam']['w'], meta['cam']['h'])).permute(1, 0).view(kpt2d.size(0), 1, 2).to(device)
        kpt2d *= img_size
        kpt2d_pred *= img_size

        epe_dist = epe(kpt2d-kpt2d[:,:1,:], kpt2d_pred - kpt2d_pred[:,:1,:])
        p_epe_dist = p_epe(kpt2d.clone(), kpt2d_pred.clone())
        epe_error.update(epe_dist.item(), kpt2d.size(0))
        p_epe_error.update(p_epe_dist.item(), kpt2d.size(0))
    print(f'epe Distance : {epe_error.avg: .8f}')
    print(f'p_epe Distance : {p_epe_error.avg: .8f}')

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch: {epoch + 1} | LR: {lr_scheduler.get_last_lr()[0]: .6f}')

        epoch_loss_2d_pos = AverageMeter()
        self_sup.train()
        bar = tqdm(train_loader, desc='[TRAIN]', total=len(train_loader))
        for kpt2d, _, meta in bar:
            kpt2d = kpt2d[:,:,:2].to(device)
            kpt2d -= kpt2d[:,:1,:]
            kpt2d, _ = align_bone_xaxis(kpt2d.clone())
            optimizer.zero_grad()
            bool_mask, recov_kpt2d, _ = self_sup(kpt2d)
            # calculate reconstruction loss
            loss = kpt2d_recon_loss(recov_kpt2d, kpt2d, target_weight=bool_mask)
            if args.half:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if args.max_norm:
                nn.utils.clip_grad_norm_(self_sup.parameters(), max_norm=1)
            optimizer.step()
            lr_scheduler.step()
            epoch_loss_2d_pos.update(loss.item(), kpt2d.size(0))
            bar.set_description(
                f"[TRAIN] Loss: {epoch_loss_2d_pos.avg: .6f}",
                refresh=True
            )

    ## hyperparameter search
    x_values, y_values_p_epe, y_values_epe = [], [], []
    for thd in range(args.thd_percent_st, args.thd_percent_st + args.thd_percent_offset, args.thd_percent_stride):
        with torch.no_grad():
            val_epoch_refine_2d_pos_epe = AverageMeter()
            val_epoch_refine_2d_pos_p_epe = AverageMeter()
            self_sup.eval()
            bar = tqdm(val_loader, desc='[EVAL]', total=len(val_loader))
            for gt, kpt2d_pred, meta in bar:
                kpt2d = gt[:,:,:2].to(device)
                kpt2d_pred = kpt2d_pred.to(device)
                img_size = torch.stack((meta['cam']['w'], meta['cam']['h'])).permute(1, 0).view(kpt2d.size(0), 1, 2).to(device)

                score = kpt2d_pred[:, :, 2].clone()
                mask_cond = score < percentile[thd]
                kpt2d_pred = kpt2d_pred[:, :, :2]
                kpt_root = kpt2d_pred[:, :1, :].clone()
                kpt2d_pred_localized = kpt2d_pred - kpt_root

                kpt2d_pred_rotated, rot_inv = align_bone_xaxis(kpt2d_pred_localized.clone())
                x = kpt2d_pred_rotated
                bool_mask, recov_kpt2d, _ = self_sup(x, mask_cond)
                # calculate reconstruction loss
                recov_kpt2d = torch.matmul(recov_kpt2d, rot_inv)
                recon_kpt2d = torch.where(bool_mask.unsqueeze(-1), recov_kpt2d.double(), kpt2d_pred_localized)

                epe_dist = epe(
                    (kpt2d - kpt2d[:,:1,:]) * img_size,
                    (recon_kpt2d) * img_size,
                )
                p_epe_dist = p_epe(
                    kpt2d * img_size,
                    (recon_kpt2d + kpt_root) * img_size,
                )

                val_epoch_refine_2d_pos_epe.update(epe_dist.item(), kpt2d.size(0))
                val_epoch_refine_2d_pos_p_epe.update(p_epe_dist.item(), kpt2d.size(0))
            print(f"[EVAL/{thd}% percentile, thd:{percentile[thd]}] EPE_dist : {val_epoch_refine_2d_pos_epe.avg: .2f} |{epe_error.avg: .2f}")
            print(f"[EVAL/{thd}% percentile, thd:{percentile[thd]}] P-EPE_dist : {val_epoch_refine_2d_pos_p_epe.avg: .2f} |{p_epe_error.avg: .2f}")
            x_values.append(thd)
            y_values_p_epe.append(val_epoch_refine_2d_pos_p_epe.avg)
            y_values_epe.append(val_epoch_refine_2d_pos_epe.avg)
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values_epe)
    ax2 = ax.twinx()
    ax2.plot(x_values, y_values_p_epe, color='red')
    plt.gcf().set_dpi(300)
    ax.set_xlabel('Confidence percentile')
    ax.set_ylabel('EPE')
    ax2.set_ylabel('P-EPE')
    plt.title(str(args.mask_prob*100)+"% masking, " + str(args.proj_kernels))
    save_ckpt({
        'state_dict': self_sup.encoder.state_dict(),
        'mask_token': self_sup.mask_token,
        'epoch': args.epochs + 1
    }, ckpt_dir_path, suffix='best')

if __name__ == '__main__':
    args = get_parse_args()

    init(args.random_seed)
    main(args)

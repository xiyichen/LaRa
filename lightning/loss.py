import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from torch.nn import functional as F
import pdb

from torch.cuda.amp import autocast
from torchmetrics.functional.regression import pearson_corrcoef

class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()

        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, batch, output, iter):

        scalar_stats = {}
        loss = 0

        B,V,H,W = batch['tar_rgb'].shape[:-1]

        tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)
        
        if 'image' in output:

            for prex in ['','_fine']:
                
                if self.training:
                    rendered_depth = output['depth'][...,0].clone()
                    masks = batch['tar_msk']
                    near_far = batch['near_far']
                    sapiens_depth = batch['tar_depths'].permute(0,2,1,3).reshape(B,H,V*W)
                    loss_depth = 0
                    for bn in range(B):
                        for vn in range(V):
                            # near, far = near_far[bn][vn]
                            mask = masks[bn][vn]
                            # rendered_depth[bn,:,vn*512:(vn+1)*512][mask==0] = 0
                            # depth_slice = rendered_depth[bn,:,vn*512:(vn+1)*512][mask==1]
                            # pdb.set_trace()
                            # depth_slice = torch.clamp(depth_slice, near, far)
                            # depth_slice = (depth_slice - depth_slice.min()) / (depth_slice.max() - depth_slice.min())
                            # rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1] = depth_slice
                            # rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1] = (rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1] - rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1].min()) / (rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1].max() - rendered_depth[bn, :, vn*512:(vn+1)*512][mask == 1].min())
                            # loss_depth += ((1 - pearson_corrcoef(sapiens_depth[bn, :, vn*512:(vn+1)*512][mask==1], depth_slice)) + ((rendered_depth[bn,:,vn*512:(vn+1)*512][mask==0])**2).mean())
                            sapiens_depth_view = sapiens_depth[bn, :, vn*512:(vn+1)*512][mask == 1].reshape(-1,1)
                            rendered_depth_view = rendered_depth[bn, :,vn*512:(vn+1)*512][mask == 1].reshape(-1,1)
                            rendered_depth_view = torch.clamp(rendered_depth_view, torch.quantile(rendered_depth_view, 0.005), torch.quantile(rendered_depth_view, 0.995))
                            
                            loss_depth += min(1 - pearson_corrcoef(sapiens_depth_view, rendered_depth_view), 1 - pearson_corrcoef(1/(sapiens_depth_view+200), rendered_depth_view))
                    loss_depth /= (B*V)
                    loss += loss_depth * 0.2
                    scalar_stats.update({f'depth{prex}': loss_depth.detach()})
                
                
                if prex=='_fine' and f'acc_map{prex}' not in output:
                    continue

                color_loss_all = (output[f'image{prex}']-tar_rgb)**2
                loss += color_loss_all.mean()

                psnr = -10. * torch.log(color_loss_all.detach().mean()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss_all.device))
                scalar_stats.update({f'mse{prex}': color_loss_all.mean().detach()})
                scalar_stats.update({f'psnr{prex}': psnr})


                with autocast(enabled=False): 
                    ssim_val = self.ssim(output[f'image{prex}'].permute(0,3,1,2), tar_rgb.permute(0,3,1,2))
                    scalar_stats.update({f'ssim{prex}': ssim_val.detach()})
                    loss += 0.5 * (1-ssim_val)
                
                if f'rend_dist{prex}' in output and iter>0 and prex!='_fine':
                    distortion = output[f"rend_dist{prex}"].mean()
                    scalar_stats.update({f'distortion{prex}': distortion.detach()})
                    loss += distortion*1000
                    
                    rend_normal  = output[f'rend_normal{prex}']
                    depth_normal = output[f'depth_normal{prex}']
                    acc_map = output[f'acc_map{prex}'].detach()

                    normal_error = ((1 - (rend_normal * depth_normal).sum(dim=-1))*acc_map).mean() 
                    scalar_stats.update({f'normal{prex}': normal_error.detach()})
                    loss += normal_error*0.2
     
        return loss, scalar_stats


import numpy as np
from glob import glob
import random
import torch
from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R
from dataLoader.SMCReader import SMCReader
import cv2
from scipy.ndimage import label
import pdb
import h5py
from smplx.body_models import SMPLX
import torch.nn.functional as F

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def visual_hull_samples(masks, KRT, n_pts=100_000, grid_resolution=256, aabb=(-1., 1.)):
    """ 
    Args:
        masks: (n_images, H, W)
        KRT: (n_images, 3, 4)
        grid_resolution: int
        aabb: (2)
    """
    # create voxel grid coordinates
    grid = np.linspace(aabb[0], aabb[1], grid_resolution)
    grid = np.meshgrid(grid, grid, grid)
    grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

    # project grid locations to the image plane
    grid = np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1) # n_pts, 4
    grid = grid[None].repeat(masks.shape[0], axis=0) # n_imgs, n_pts, 4
    grid = grid @ KRT.transpose(0, 2, 1)  # (n_imgs, n_pts, 4) @ (n_imgs, 4, 3) -> (n_imgs, n_pts, 3)
    uv = grid[..., :2] / grid[..., 2:] # (n_imgs, n_pts, 2)
    _, H, W = masks.shape[:3]  # n_imgs,H,W
    uv[..., 0] = 2.0 * (uv[..., 0] / (W - 1.0)) - 1.0
    uv[..., 1] = 2.0 * (uv[..., 1] / (H - 1.0)) - 1.0

    uv = torch.from_numpy(uv).float()
    masks = torch.from_numpy(masks)[:, None].squeeze(-1).float()
    samples = F.grid_sample(masks, uv[:, None], align_corners=True, mode='nearest', padding_mode='zeros').squeeze()
    _ind = (samples > 0).all(0) # (n_imgs, n_pts) -> (n_pts)

    # sample points around the grid locations
    grid_samples = grid_loc[_ind] # (n_pts, 2)
    all_samples = grid_samples
    np.random.shuffle(all_samples)

    return all_samples[:n_pts]


def visual_hull_samples_list(masks_list, KRT, n_pts=100000, grid_resolution=256, aabb=(-1, 1)):
    """ Visual hull from multiple views (images of different resolutions)
    Args:
        masks_list (list): n_images images of (H, W)
        KRT: (n_images, 3, 4)
        grid_resolution: int
        aabb: (2)
    """
    # create voxel grid coordinates
    grid = np.linspace(aabb[0], aabb[1], grid_resolution)
    grid = np.meshgrid(grid, grid, grid)
    grid_loc = np.stack(grid, axis=-1).reshape(-1, 3) # n_pts, 3

    # project grid locations to the image plane
    grid = np.concatenate([grid_loc, np.ones_like(grid_loc[:, :1])], axis=-1) # n_pts, 4
    grid = grid[None].repeat(KRT.shape[0], axis=0) # n_imgs, n_pts, 4
    grid = grid @ KRT.transpose(0, 2, 1)  # (n_imgs, n_pts, 4) @ (n_imgs, 4, 3) -> (n_imgs, n_pts, 3)
    uv = grid[..., :2] / grid[..., 2:] # (n_imgs, n_pts, 2)
    # pdb.set_trace()
    for i in range(len(masks_list)):
        H, W = masks_list[i].shape[:2]
        uv[i, ..., 0] = 2.0 * (uv[i, ..., 0] / (W - 1.0)) - 1.0
        uv[i, ..., 1] = 2.0 * (uv[i, ..., 1] / (H - 1.0)) - 1.0

    uv = torch.from_numpy(uv).float()
    outside_pts = (uv[..., 0] < -1) | (uv[..., 0] > 1) | (uv[..., 1] < -1) | (uv[..., 1] > 1)
    samples = []
    for i in range(len(masks_list)):
        mask = torch.from_numpy(masks_list[i][None])[:, None].squeeze(-1).float()
        samples.append(F.grid_sample(mask, uv[i][None][:, None], align_corners=True, mode='nearest', padding_mode='zeros').squeeze())
    samples = torch.stack(samples, dim=0)
    samples = samples > 0
    # samples = samples | outside_pts
    # pdb.set_trace()
    _ind = samples.all(0) # (n_imgs, n_pts) -> (n_pts)

    # sample points around the grid locations
    grid_samples = grid_loc[_ind] # (n_pts, 2)
    all_samples = grid_samples
    np.random.shuffle(all_samples)

    return all_samples[:n_pts]

def recenter(image, mask, border_ratio = 0.2):
    """ recenter an image to leave some empty space at the image border.

    Args:
        image (ndarray): input image, float/uint8 [H, W, 3/4]
        mask (ndarray): alpha mask, bool [H, W]
        border_ratio (float, optional): border ratio, image will be resized to (1 - border_ratio). Defaults to 0.2.

    Returns:
        ndarray: output image, float/uint8 [H, W, 3/4]
    """
    
    H, W, C = image.shape
    size = max(H, W)

    # default to white bg if rgb, but use 0 if rgba
    if C == 3:
        result = np.ones((size, size, C), dtype=np.float32)*255
    else:
        result = np.zeros((size, size, C), dtype=np.float32)
        result[:,:,:3] = 255
            
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    
    h = x_max - x_min
    w = y_max - y_min
    desired_size = int(size * (1 - border_ratio))
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    x2_min = (size - h2) // 2
    x2_max = x2_min + h2
    y2_min = (size - w2) // 2
    y2_max = y2_min + w2
    result[x2_min:x2_max, y2_min:y2_max] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    
    return result, x_min, y_min, h2/(x_max-x_min), w2/(y_max-y_min), x2_min, y2_min

def load_cameras(annot_reader, cam_ids):
    # Load K, R, T
    cameras = {'K': [], 'R': [], 'T': [], 'D': []}
    for i in range(16):
        cam_params = annot_reader.get_Calibration(cam_ids[i])
        K = cam_params['K']
        D = cam_params['D'] # k1, k2, p1, p2, k3
        c2w = cam_params['RT']
        RT = np.linalg.inv(c2w)
        cameras['K'].append(K)
        cameras['R'].append(RT[:3,:3])
        cameras['T'].append(RT[:3,3].reshape(3,))
        cameras['D'].append(D)
    for k in cameras:
        cameras[k] = np.stack(cameras[k], axis=0)
    
    return cameras

def get_front_view_idx(annot_reader, cameras, frame_idx):
    kps_3d = annot_reader.get_Keypoints3d(frame_idx)[:,:3]
    angle_degs = []
    for cam_idx in range(len(cameras['K'])):
        kps_3d_cam = (cameras['R'][cam_idx]@kps_3d.T).T+cameras['T'][cam_idx].reshape(1,3)
        
        p1 = (kps_3d_cam[16] + kps_3d_cam[17])/2
        p2 = kps_3d_cam[1]
        p3 = kps_3d_cam[2]
        v1 = p2 - p1
        v2 = p3 - p1

        # Compute the cross product
        plane_normal = np.cross(v1, v2)
        plane_normal = plane_normal/np.linalg.norm(plane_normal)
        negative_z_axis = np.array([0, 0, 1])
        
        dot_product = np.dot(plane_normal, negative_z_axis)
        angle_rad = np.arccos(dot_product)  # Angle in radians
        angle_deg = np.degrees(angle_rad)  # Angle in degrees
        angle_degs.append(angle_deg)
    
    return angle_degs.index(min(angle_degs))

class HumanDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        super(HumanDataset, self).__init__()
        self.img_size = np.array(cfg.img_size)
            
        self.scenes_name = ['0008_01']
        # self.scenes_name = ['0019_06']
        # self.scenes_name = ['0018_05']
        # self.scenes_name = ['0012_09']
        # self.scenes_name = ['0811_08']

    def __getitem__(self, index):
        file_id = self.scenes_name[index]
        main_file = f'/fs/gamma-datasets/MannequinChallenge/dna_rendering_data/{file_id}.smc'
        annot_file = main_file.replace('.smc', '_annots.smc')
        main_reader = SMCReader(main_file)
        annot_reader = SMCReader(annot_file)
        frame_idx = 0
        cam_ids_low = [2,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]
        cameras_low = load_cameras(annot_reader, cam_ids_low)
        
        front_view_idx = get_front_view_idx(annot_reader, cameras_low, frame_idx)
        left_view_idx = (front_view_idx+4)%16
        back_view_idx = (front_view_idx+8)%16
        right_view_idx = (front_view_idx+12)%16
        
        bg_white = np.array([255, 255, 255])
        tar_ixts = []
        Ks = []
        tar_w2cs = []
        tar_img = []
        tar_msks = []
        for cam_idx in ([front_view_idx, left_view_idx, back_view_idx, right_view_idx]):
            # Load image, mask
            image = main_reader.get_img('Camera_5mp', cam_ids_low[cam_idx], Image_type='color', Frame_id=frame_idx)
            image = cv2.undistort(image, cameras_low['K'][cam_idx], cameras_low['D'][cam_idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            mask = annot_reader.get_mask(cam_ids_low[cam_idx], Frame_id=frame_idx)
            mask = cv2.undistort(mask, cameras_low['K'][cam_idx], cameras_low['D'][cam_idx])
            mask = mask[..., np.newaxis].astype(np.float32) / 255.0
            # filter out small floating parts in the mask
            labeled_mask, _ = label(mask)
            component_sizes = np.bincount(labeled_mask.ravel())
            component_sizes[0] = 0
            largest_component_label = component_sizes.argmax()
            mask = (labeled_mask == largest_component_label)
            
            image = image * mask + bg_white * (1.0 - mask)
            K = cameras_low['K'][cam_idx].copy()
            
            rgba = np.zeros((image.shape[0], image.shape[1], 4))
            rgba[:,:,:3] = image
            rgba[:,:,-1:] = mask.astype(np.float32)*255
            
            rgba, x1, y1, s1, s2, x2, y2 = recenter(rgba, mask.astype(np.float32)*255, border_ratio=0.05)
            image = rgba[:,:,:3]
            mask = rgba[:,:,-1] / 255
            
            K[0][2] -= y1
            K[1][2] -= x1
            K[0] *= s2
            K[1] *= s1
            K[0][2] += y2
            K[1][2] += x2
            K *= (512/image.shape[0])
            K[-1,-1] = 1
            image = cv2.resize(image, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            mask[mask<0.5] = 0
            mask[mask>=0.5] = 1
            # cv2.imwrite('./debug.png', image[:,:,::-1])
            # cv2.imwrite('./debug_mask.png', mask*255)
            # pdb.set_trace()
            Ks.append(K)
            K_pseudo = np.eye(3)
            K_pseudo[0][0] = 955.4050
            K_pseudo[1][1] = 955.4050
            K_pseudo[0][2] = 256
            K_pseudo[1][2] = 256
            tar_ixts.append(K_pseudo)
            
            w2c_lgm = np.eye(4)
            w2c_lgm[:3,:3] = cameras_low['R'][cam_idx]
            w2c_lgm[:3,3] = cameras_low['T'][cam_idx].reshape(3,)
            tar_w2cs.append(w2c_lgm)
            tar_img.append(image/255)
            tar_msks.append(mask)
        tar_img = np.stack(tar_img, axis=0)
        tar_img = torch.from_numpy(tar_img).clamp(0,1).float()
        tar_msks = np.stack(tar_msks, axis=0)
        tar_msks = torch.from_numpy(tar_msks).clamp(0,1).float()
        tar_w2cs = np.stack(tar_w2cs, axis=0)
        tar_ixts = np.stack(Ks, axis=0)
        bg_colors = torch.ones(4,3).float()
        
        sampled_points = visual_hull_samples(tar_msks.detach().cpu().numpy().astype(np.float32), tar_ixts@tar_w2cs[:,:3,:4].astype(np.float32))

        center = (sampled_points.min(axis=0) + sampled_points.max(axis=0))/2
        
        sampled_points -= center
        tar_w2cs[:,:3,3] += (tar_w2cs[:,:3,:3]@center.reshape(3,1)).reshape(-1,3)
        tar_w2cs[:,:3,3] *= (0.4 / sampled_points.max(axis=0).max())
        
        
        R = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        tar_w2cs[:,:3,:3] = tar_w2cs[:,:3,:3]@R[:3,:3].T[None]
        
        # get tar_c2ws
        tar_c2ws = np.linalg.inv(tar_w2cs)

        # align cameras using first view
        # no inverse operation 
        r = np.linalg.norm(tar_c2ws[0,:3,3])
        ref_c2w = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_w2c = np.eye(4, dtype=np.float32).reshape(1,4,4)
        ref_c2w[:,2,3], ref_w2c[:,2,3] = -r, r
        transform_mats = ref_c2w @ tar_w2cs[:1]
        tar_w2cs = tar_w2cs.copy() @ tar_c2ws[:1] @ ref_w2c
        tar_c2ws = transform_mats @ tar_c2ws.copy()
        
        ret = {}
        H, W = self.img_size

        ret.update({'tar_c2w': tar_c2ws,
                    'tar_w2c': tar_w2cs,
                    'tar_ixt': tar_ixts,
                    'tar_rgb': tar_img,
                    'tar_msk': tar_msks,
                    'transform_mats': transform_mats,
                    'bg_color': bg_colors
                    })
        
        near_far = np.array([r-0.8, r+0.8]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': 'human', 'tar_view': [0,1,2,3], 'frame_id': 0}})
        ret['meta'].update({f'tar_h': int(H), f'tar_w': int(W)})

        rays = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0)
        ret.update({f'tar_rays': rays})
        rays_down = build_rays(tar_c2ws, tar_ixts.copy(), H, W, 1.0/16)
        ret.update({f'tar_rays_down': rays_down})
        return ret
    
    def read_views(self, scene, src_views, scene_name):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, msks, normals = [], [], [], [], [], []
        for i, idx in enumerate(src_ids):
            
            if self.split!='train' or i < self.n_group:
                bg_color = np.ones(3).astype(np.float32)
            else:
                bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])

            bg_colors.append(bg_color)
            
            img, normal, mask = self.read_image(scene, idx, bg_color, scene_name)
            imgs.append(img)
            ixt, ext, w2c = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
        return np.stack(imgs), np.stack(bg_colors), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)

    def read_cam(self, scene, view_idx):
        c2w = np.array(scene[f'c2w_{view_idx}'], dtype=np.float32)
        w2c = np.linalg.inv(c2w)
        fov = np.array(scene[f'fov_{view_idx}'], dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)
        return ixt, c2w, w2c

    def read_image(self, scene, view_idx, bg_color, scene_name):
        
        img = np.array(scene[f'image_{view_idx}'])

        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)

        if self.cfg.load_normal:

            normal = np.array(scene[f'normal_{view_idx}'])
            normal = normal.astype(np.float32) / 255. * 2 - 1.0
            return img, normal, mask

        return img, None, mask


    def __len__(self):
        return len(self.scenes_name)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K


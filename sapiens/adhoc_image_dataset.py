# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2
import pdb

import h5py
import cv2
import numpy as np 
from tqdm import tqdm 
import pdb

class SMCReader:

    def __init__(self, file_path):
        """Read SenseMocapFile endswith ".smc".

        Args:
            file_path (str):
                Path to an SMC file.
            body_model (nn.Module or dict):
                Only needed for SMPL transformation to device frame
                if nn.Module: a body_model instance
                if dict: a body_model config
        """
        self.smc = h5py.File(file_path, 'r')
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None 
        self.__available_keys__ = list(self.smc.keys())
        
        self.actor_info = None 
        if hasattr(self.smc, 'attrs') and len(self.smc.attrs.keys()) > 0:
            self.actor_info = dict(
                id=self.smc.attrs['actor_id'],
                perf_id=self.smc.attrs['performance_id'],
                age=self.smc.attrs['age'],
                gender=self.smc.attrs['gender'],
                height=self.smc.attrs['height'],
                weight=self.smc.attrs['weight'],
                # ethnicity=self.smc.attrs['ethnicity'],
            )

        self.Camera_5mp_info = None 
        if 'Camera_5mp' in self.smc:
            self.Camera_5mp_info = dict(
                num_device=self.smc['Camera_5mp'].attrs['num_device'],
                num_frame=self.smc['Camera_5mp'].attrs['num_frame'],
                resolution=self.smc['Camera_5mp'].attrs['resolution'],
            )
        self.Camera_12mp_info = None 
        if 'Camera_12mp' in self.smc:
            self.Camera_12mp_info = dict(
                num_device=self.smc['Camera_12mp'].attrs['num_device'],
                num_frame=self.smc['Camera_12mp'].attrs['num_frame'],
                resolution=self.smc['Camera_12mp'].attrs['resolution'],
            )
        self.Kinect_info = None
        if 'Kinect' in self.smc:
            self.Kinect_info=dict(
                num_device=self.smc['Kinect'].attrs['num_device'],
                num_frame=self.smc['Kinect'].attrs['num_frame'],
                resolution=self.smc['Kinect'].attrs['resolution'],
            )

    def get_available_keys(self):
        return self.__available_keys__ 

    def get_actor_info(self):
        return self.actor_info
    
    def get_Camera_12mp_info(self):
        return self.Camera_12mp_info

    def get_Camera_5mp_info(self):
        return self.Camera_5mp_info
    
    def get_Kinect_info(self):
        return self.Kinect_info
    
    ### RGB Camera Calibration
    def get_Calibration_all(self):
        """Get calibration matrix of all cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_Parameter: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_id(str) in {'Camera_5mp': '0'~'47',  'Camera_12mp':'48'~'60'}
                Matrix_type in ['D', 'K', 'RT', 'Color_Calibration'] 
        """  
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        if self.__calibration_dict__ is not None:
            return self.__calibration_dict__

        self.__calibration_dict__ = dict()
        for ci in self.smc['Camera_Parameter'].keys():
            self.__calibration_dict__.setdefault(ci,dict())
            for mt in ['D', 'K', 'RT', 'Color_Calibration'] :
                self.__calibration_dict__[ci][mt] = \
                    self.smc['Camera_Parameter'][ci][mt][()]
        return self.__calibration_dict__

    def get_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain camera by its type and id 

        Args:
            Camera_id (int/str of a number):
                Camera_id(str) in {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT', 'Color_Calibration'] 
        """
        if not 'Camera_Parameter' in self.smc:
            print("=== no key: Camera_Parameter.\nplease check available keys!")
            return None  

        rs = dict()
        for k in ['D', 'K', 'RT', 'Color_Calibration'] :
            rs[k] = self.smc['Camera_Parameter'][f'{int(Camera_id):02d}'][k][()]
        return rs

    ### Kinect Camera Calibration
    def get_Kinect_Calibration_all(self):
        """Get calibration matrix of all kinect cameras and save it in self
        
        Args:
            None

        Returns:
            Dictionary of calibration matrixs of all matrixs.
              dict( 
                Camera_group: Camera_id : Matrix_type : value
              )
            Notice:
                Camera_group(str) in ['Kinect']
                Camera_id(str) in {'Kinect': '0'~'7'}
                Matrix_type in ['D', 'K', 'RT'] 
        """  
        if not 'Calibration' in self.smc:
            print("=== no key: Calibration.\nplease check available keys!")
            return None  

        if self.__kinect_calib_dict__ is not None:
            return self.__kinect_calib_dict__

        self.__kinect_calib_dict__ = dict()
        for cg in ['Kinect']:
            self.__kinect_calib_dict__.setdefault(cg,dict())
            for ci in self.smc['Calibration'][cg].keys():
                self.__kinect_calib_dict__[cg].setdefault(ci,dict())
                for mt in ['D', 'K', 'RT'] :
                    self.__kinect_calib_dict__[cg][ci][mt] = \
                        self.smc['Calibration'][cg][ci][mt][()]
        return self.__kinect_calib_dict__

    def get_kinect_Calibration(self, Camera_id):
        """Get calibration matrixs of a certain kinect camera by its type and id 

        Args:
            Camera_group (str):
                Camera_group in ['Kinect'].
            Camera_id (int/str of a number):
                CameraID(str) in {'Kinect': '0'~'7'}
        Returns:
            Dictionary of calibration matrixs.
                ['D', 'K', 'RT'] 
        """  
        if not 'Calibration' in self.smc:
            print("=== no key: Calibration.\nplease check available keys!")
            return None 

        Camera_id = f'{int(Camera_id):02d}'
        assert(Camera_id in self.smc['Calibration']["Kinect"].keys())
        rs = dict()
        for k in ['D', 'K', 'RT']:
            rs[k] = self.smc['Calibration']["Kinect"][Camera_id][k][()]
        return rs

    ### RGB image
    def __read_color_from_bytes__(self, color_array):
        """Decode an RGB image from an encoded byte array."""
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    def get_mask(self, Camera_id, Frame_id=None,disable_tqdm=True):
        """Get mask from Camera_id, Frame_id

        Args:
            Camera_id (int/str of a number):
                Camera_id (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not 'Mask' in self.smc:
            print("=== no key: Mask.\nplease check available keys!")
            return None  

        Camera_id = str(Camera_id)

        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc['Mask'][Camera_id]['mask'].keys())
            img_byte = self.smc['Mask'][Camera_id]['mask'][Frame_id][()]
            img_color = self.__read_color_from_bytes__(img_byte)
            img_color = np.max(img_color,2)
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc['Mask'][Camera_id]['mask'].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_mask(Camera_id,fi))
            return np.stack(rs,axis=0)

    def get_img(self, Camera_group, Camera_id, Image_type,Frame_id=None,disable_tqdm=True):
        """Get image its Camera_group, Camera_id, Image_type and Frame_id

        Args:
            Camera_group (str):
                Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'].
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',
                    'Kinect': '0'~'7'}
            Image_type(str) in 
                    {'Camera_5mp': ['color'],  
                    'Camera_12mp': ['color'],
                    'Kinect': ['depth', 'mask']}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not Camera_group in self.smc:
            print("=== no key: %s.\nplease check available keys!" % Camera_group)
            return None

        assert(Camera_group in ['Camera_12mp', 'Camera_5mp','Kinect'])
        Camera_id = str(Camera_id)
        assert(Camera_id in self.smc[Camera_group].keys())
        assert(Image_type in self.smc[Camera_group][Camera_id].keys())
        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = str(Frame_id)
            assert(Frame_id in self.smc[Camera_group][Camera_id][Image_type].keys())
            if Image_type in ['color']:
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
            if Image_type == 'mask':
                img_byte = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
                img_color = self.__read_color_from_bytes__(img_byte)
                img_color = np.max(img_color,2)
            if Image_type == 'depth':
                img_color = self.smc[Camera_group][Camera_id][Image_type][Frame_id][()]
            return img_color           
        else:
            if Frame_id is None:
                Frame_id_list =sorted([int(l) for l in self.smc[Camera_group][Camera_id][Image_type].keys()])
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm(Frame_id_list, disable=disable_tqdm):
                rs.append(self.get_img(Camera_group, Camera_id, Image_type,fi))
            return np.stack(rs,axis=0)
    
    ###Keypoints2d
    def get_Keypoints2d(self, Camera_id, Frame_id=None):
        """Get keypoint2D by its Camera_group, Camera_id and Frame_id

        Args:
            Camera_id (int/str of a number):
                CameraID (str) in 
                    {'Camera_5mp': '0'~'47',  
                    'Camera_12mp':'48'~'60',}
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            a single img :
              'color': HWC in bgr (uint8)
              'mask' : HW (uint8)
              'depth': HW (uint16)
        """ 
        if not 'Keypoints_2D' in self.smc:
            print("=== no key: Keypoints_2D.\nplease check available keys!")
            return None 

        Camera_id = f'{int(Camera_id):02d}'
        assert(isinstance(Frame_id,(list,int, str, type(None))))
        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_2D'][Camera_id][()][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_2D'][Camera_id][()]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in  tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints2d(Camera_id,fi))
            return np.stack(rs,axis=0)

    ###Keypoints3d
    def get_Keypoints3d(self, Frame_id=None):
        """Get keypoint3D Frame_id, TODO coordinate

        Args:
            Frame_id a.(int/str of a number): '0' ~ 'num_frame-1'('149') 
                     b.list of numbers (int/str)
                     c.None: get batch of all imgs in order of time sequence 
        Returns:
            Keypoints3d tensor: np.ndarray of shape ([N], ,3)
        """ 
        if not 'Keypoints_3D' in self.smc:
            print("=== no key: Keypoints_3D.\nplease check available keys!")
            return None 

        if isinstance(Frame_id, (str,int)):
            Frame_id = int(Frame_id)
            return self.smc['Keypoints_3D']["keypoints3d"][Frame_id,:]
        else:
            if Frame_id is None:
                return self.smc['Keypoints_3D']["keypoints3d"]
            elif isinstance(Frame_id, list):
                Frame_id_list = Frame_id
            rs = []
            for fi in tqdm.tqdm(Frame_id_list):
                rs.append(self.get_Keypoints3d(fi))
            return np.stack(rs,axis=0)

    ###SMPLx
    def get_SMPLx(self, Frame_id=None):
        """Get SMPL (world coordinate) computed by mocap processing pipeline.

        Args:
            Frame_id (int, list or None, optional):
                int: frame id of one selected frame
                list: a list of frame id
                None: all frames will be returned
                Defaults to None.

        Returns:
            dict:
                'global_orient': np.ndarray of shape (N, 3)
                'body_pose': np.ndarray of shape (N, 21, 3)
                'transl': np.ndarray of shape (N, 3)
                'betas': np.ndarray of shape (1, 10)
        """
        if not 'SMPLx' in self.smc:
            print("=== no key: SMPLx.\nplease check available keys!")
            return None 

        t_frame = self.smc['SMPLx']['betas'][()].shape[0]
        print("=== t_frame", self.smc['SMPLx']['betas'][()].shape, flush=True)

        key_arr = ['betas', 'expression', 'fullpose', 'transl']
        for k in key_arr:
            print("=== smplx %s" % k, self.smc['SMPLx'][k][()].shape[0], flush=True)
        
        if Frame_id is None:
            frame_list = range(t_frame)
        elif isinstance(Frame_id, list):
            frame_list = [int(fi) for fi in Frame_id]
        elif isinstance(Frame_id, (int,str)):
            Frame_id = int(Frame_id)
            assert Frame_id < t_frame,\
                f'Invalid frame_index {Frame_id}'
            frame_list = Frame_id
        else:
            raise TypeError('frame_id should be int, list or None.')

        smpl_dict = {}
        for key in ['betas', 'expression', 'fullpose', 'transl']:
            smpl_dict[key] = self.smc['SMPLx'][key][()][frame_list, ...]
        smpl_dict['scale'] = self.smc['SMPLx']['scale'][()]

        return smpl_dict

    def release(self):
        self.smc = None 
        self.__calibration_dict__ = None
        self.__kinect_calib_dict__ = None
        self.__available_keys__ = None
        self.actor_info = None 
        self.Camera_5mp_info = None
        self.Camera_12mp_info = None 
        self.Kinect_info = None


def load_cameras(annot_reader, cam_ids):
    # Load K, R, T
    cameras = {'K': [], 'R': [], 'T': [], 'D': []}
    for i in range(len(cam_ids)):
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

class AdhocImageDataset(torch.utils.data.Dataset):
    def __init__(self, file_ids, shape=None, mean=None, std=None, index=0):
        # self.image_list = image_list
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        # file_id = '0047_12'
        self.file_ids = file_ids
        self.cameras = {}
        self.main_readers = {}
        self.annot_readers = {}
        
        self.len = 0
        self.metadata = []
        every_n_frames = 20
        for file_idx, file_id in enumerate(file_ids):
            if file_idx < index*50 or file_idx >= (index+1)*50:
                continue
            main_file = f'/fs/gamma-datasets/MannequinChallenge/dna_rendering_data/{file_id}.smc'
            annot_file = main_file.replace('.smc', '_annots.smc')
            self.main_readers[file_id] = SMCReader(main_file)
            self.annot_readers[file_id] = SMCReader(annot_file)
            num_frames = int(self.main_readers[file_id].smc['Camera_5mp'].attrs['num_frame'])//every_n_frames
            self.cameras[file_id] = load_cameras(self.annot_readers[file_id], [2,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5])
            for cam_idx, cam_id in enumerate([2,47,44,41,38,35,32,29,26,23,20,17,14,11,8,5]):
                self.len += num_frames
                for frame_idx in range(num_frames):
                    self.metadata.append([file_id, cam_idx, cam_id, frame_idx*every_n_frames])

    def __len__(self):
        return self.len
    
    def _preprocess(self, img):
        if self.shape:
            img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img
    
    def __getitem__(self, idx):
        # orig_img_dir = self.image_list[idx]
        # orig_img = cv2.imread(orig_img_dir)
        # # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        # img = self._preprocess(orig_img)
        # return orig_img_dir, orig_img, img
        
        
        # file_idx = idx // 60
        # file_id = self.file_ids[file_idx]
        # cam_idx = idx - file_idx*60
        # frame_idx = 0
        # if cam_idx < 48:
        #     orig_img = self.main_readers[file_id].get_img('Camera_5mp', str(cam_idx), Image_type='color', Frame_id=frame_idx)
        # else:
        #     orig_img = self.main_readers[file_id].get_img('Camera_12mp', str(cam_idx), Image_type='color', Frame_id=frame_idx)
        # orig_img = cv2.undistort(orig_img, self.cameras[file_id]['K'][cam_idx], self.cameras[file_id]['D'][cam_idx])
        # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # img = self._preprocess(orig_img)
        # # print(file_id, str(cam_idx) + '.png', orig_img.shape, img.shape)
        # return file_id, str(cam_idx) + '.png', orig_img, img
        
        metadata = self.metadata[idx]
        file_id, cam_idx, cam_id, frame_idx = metadata
        orig_img = self.main_readers[file_id].get_img('Camera_5mp', str(cam_id), Image_type='color', Frame_id=frame_idx)
        orig_img = cv2.undistort(orig_img, self.cameras[file_id]['K'][cam_idx], self.cameras[file_id]['D'][cam_idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = self._preprocess(orig_img)
        # print(idx, file_id, cam_idx, cam_id, frame_idx, orig_img.shape, img.shape)
        # print(file_id, str(cam_idx) + '.png', orig_img.shape, img.shape)
        
        
        orig_mask = self.annot_readers[file_id].get_mask(str(cam_id), Frame_id=frame_idx)
        orig_mask = cv2.undistort(orig_mask, self.cameras[file_id]['K'][cam_idx], self.cameras[file_id]['D'][cam_idx]) / 255
        orig_mask[orig_mask < 0.5] = 0
        orig_mask[orig_mask >= 0.5] = 1
        return file_id, str(cam_id) + '_' + str(frame_idx) + '.png', orig_img, img, orig_mask.astype(np.bool_)
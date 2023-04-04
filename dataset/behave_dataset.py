from pathlib import Path
from dataset.base_dataset import BaseDataset
from termcolor import cprint
from configs.paths import behave_seqs_path, behave_calibs_path
import os
from os.path import exists
import numpy as np
import h5py
import torch
from PIL import Image
import re
from torchvision import transforms as transformst
from pytorch3d import transforms
from dataset import behave_camera_utils as bcu
from tqdm import tqdm
import cv2
from lib_smpl.smpl_utils import get_smplh
from pytorch3d.io import load_ply
from pytorch3d.structures import Meshes
import joblib
import smplx
from ahoi_utils import *
from visualize import *
import pickle

split = {
    "train": {"Date01", "Date02", "Date05", "Date06", "Date07"},
    "val": {"Date04"},
    "test": {"Date03"}
}

train_transformer = transformst.Compose([
        transformst.ToTensor(),
        transformst.Resize((224, 224)),
        transformst.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class BehaveDataset(BaseDataset):

    def initialize(self, opt, phase='train', cat=None):
        if phase in split:
            self.dates = split[phase]
        else:
            assert False, f"Unknown phase for BEHAVE dataset: {phase}"

        cprint(f"[*] Loading BEHAVE dataset for phase {phase}", 'yellow')
        if cat is not None:
            cprint(f"Using category {cat}", "yellow")

        self.opt = opt
        self.phase = phase
        self.cats_list = []
        self.sdf_list = []

        pbar = tqdm(desc="Number of loaded images", unit="images")
        seq_list = [seq for seq in os.listdir(
            behave_seqs_path) if seq.split("_")[0] in self.dates]

        for seq in seq_list:
            for root, _, files in os.walk(os.path.join(behave_seqs_path, seq)):
                if 'fit' not in root:
                    continue
                for f in files:
                    if re.match(r".*_fit_k\d_sdf.h5", f):
                        self.sdf_list.append(os.path.join(root, f))
                        self.cats_list.append(f.split("_")[0])
                        pbar.update(1)
        pbar.close()

        np.random.default_rng(seed=0).shuffle(self.sdf_list)
        np.random.default_rng(seed=0).shuffle(self.cats_list)

        if opt.max_dataset_size < len(self.sdf_list):
            self.sdf_list = self.sdf_list[:opt.max_dataset_size]
            self.cats_list = self.cats_list[:opt.max_dataset_size]
        # need to check the seed for reproducibility
        cprint('[*] %d samples loaded.' % (len(self.sdf_list)), 'yellow')

        self.N = len(self.sdf_list)

    def __getitem__(self, index):
        cat_name = self.cats_list[index]
        sdf_h5_file = self.sdf_list[index]

        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = torch.Tensor(sdf).view(1, 64, 64, 64)

        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)

        ret = {
            'sdf': sdf,
            'cat_str': cat_name,
            'path': sdf_h5_file,
        }

        return ret

    def __len__(self):
        return self.N

    def name(self):
        return 'BehaveDataset'

class BehaveImgDataset(BaseDataset):
    def initialize(self, opt, phase='train', cat=None, load_extra=True):
        if phase in split:
            self.dates = split[phase]
        elif phase == "test-chore":
            self.dates = split["test"]
        else:
            assert False, f"Unknown phase for BEHAVE dataset: {phase}"

        cprint(f"[*] Loading BEHAVE dataset for phase {phase}", 'yellow')
        if cat is not None:
            cprint(f"Using category {cat}", "yellow")

        self.opt = opt
        self.phase = phase
        self.data = []
        self.load_extra = load_extra

        self.intrinsics = [bcu.load_intrinsics(os.path.join(
            "dataset/calibs", "intrinsics"), i) for i in range(4)]
        
        self.camera_params = {}
        for i in range(1, 8):
            self.camera_params[f"Date0{i}"] = bcu.load_kinect_poses_back(
                os.path.join(behave_calibs_path, f"Date0{i}", "config"),
                [0,1,2,3],
                True
            )
            

        seq_list = [seq for seq in os.listdir(
            behave_seqs_path) if seq.split("_")[0] in self.dates]

        pbar = tqdm(desc="Number of loaded images", unit="images")
        for seq in seq_list:
            for root, _, files in os.walk(os.path.join(behave_seqs_path, seq)):
                if "fit" not in root:
                    continue
                for f in files:
                    if re.match(r".*_fit_k\d_sdf.h5", f):
                        
                        kid = int(f.split("_")[-2][1])
                        category = f.split("_")[0]
                        
                        h5_path = os.path.join(root, f)

                        pvqout_path = h5_path.replace("sdf.h5", "pvqout.npz")
                        obj_path = h5_path.replace("_sdf.h5", ".obj")

                        par = str(Path(h5_path).parents[2])
                        
                        img_path = os.path.join(par, f"k{kid}.color.jpg")
                        mask_path = os.path.join(
                            par, f"k{kid}.obj_rend_mask.jpg")
                        smpl_path = os.path.join(
                            par, 'person', 'fit02', 'person_fit.pkl')
                        
                        # things needed for PARE and Omni3d
                        body_mesh = None
                        if self.load_extra:
                            body_mesh = smpl_path.replace(".pkl", ".ply")

                        # filter out images with high occlusion for test-chore
                        if phase == "test-chore":
                            full_mask_path = os.path.join(
                                par, f"k{kid}.obj_rend_full.jpg")

                            if not os.path.exists(full_mask_path):
                                cprint(
                                    """Full mask not found, please download the full masks from the link below:
                                    https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-test-object-fullmask.zip
                                    Take into account that if this link doesn't work it's probably moved, please check the CHORE github repo if this happens:
                                    https://github.com/xiexh20/CHORE
                                    """,
                                    "red")
                            
                            mask = cv2.imread(
                                mask_path, cv2.IMREAD_GRAYSCALE) > 127
                            full_mask = cv2.imread(
                                full_mask_path, cv2.IMREAD_GRAYSCALE) > 127
                            if np.sum(mask) / np.sum(full_mask) < 0.3:
                                continue

                        if exists(img_path) and exists(mask_path) and exists(pvqout_path) and exists(obj_path) and category == 'backpack':
                            self.data.append({
                                'img_path': img_path,
                                'mask_path': mask_path,
                                'pvqout_path': pvqout_path,
                                'obj_path': obj_path,
                                'h5_path': h5_path,
                                'category': category,
                                'kid': kid,
                                'smpl_path': smpl_path,
                                'date': seq.split("_")[0],
                                'body_mesh': body_mesh,
                            })
                            pbar.update(1)
                        
        pbar.close()
        
        np.random.default_rng(seed=0).shuffle(self.data)

        cprint('[*] %d data loaded.' % (len(self.data)), 'yellow')
        if opt.max_dataset_size < len(self.data):
            self.data = self.data[:opt.max_dataset_size]

        self.N = len(self.data)
        self.to_tensor = transformst.ToTensor()
        #self.pare = joblib.load(open("/data/aruzzi/Behave/pare_smpl_params.pkl", 'rb'))
        self.create_smplx_model()
        self.device = torch.device('cpu')
        self.grid = torch.from_numpy(gene_voxel_grid(N=64, len=2, homo=False))[None, :]


    def __getitem__(self, index):

        data = self.data[index]
        
        path = data['h5_path']
        
        h5_file = h5py.File(path, 'r')
        
        sdf = torch.from_numpy(h5_file['pc_sdf_sample'][:].astype(np.float32))
        code = torch.from_numpy(np.load(data['pvqout_path'])[
                                'code'].astype(np.float32))
        codeix = torch.from_numpy(np.load(data['pvqout_path'])[
                                  'codeix'].astype(np.int64))

        img_path = data['img_path']
        img = Image.open(img_path).convert('RGB')
        #img = self.to_tensor(img)
        img = train_transformer(img)
        
        mask_path = data['mask_path']
        mask = Image.open(mask_path).convert('L')
        mask = self.to_tensor(mask)
        
        category = data['category']
        
        norm_params = h5_file['norm_params'][:].astype(np.float32)
        bbox = h5_file['sdf_params'][:].astype(np.float32)
        norm_params = torch.Tensor(norm_params)
        bbox = torch.Tensor(bbox).view(2, 3)
        
        bbox_scale = (bbox[1, 0]-bbox[0, 0]) * norm_params[3]
        bbox_center = (bbox[0] + bbox[1]) / 2.0 * \
            norm_params[3] + norm_params[:3]
        bbox = torch.cat([bbox_center, bbox_scale.view(1)], dim=0)

        calibration_matrix = self.intrinsics[data['kid']][0]
        dist_coefs = self.intrinsics[data['kid']][1]

        mesh_obj_path = data['obj_path']

        h5_file.close()
        
        thres = self.opt.trunc_thres
        if thres != 0.0:
            sdf = torch.clamp(sdf, min=-thres, max=thres)
            
        rt = (
            self.camera_params[data['date']][0][data['kid']],
            self.camera_params[data['date']][1][data['kid']],
        )

        # print(data['date'], data['kid'], rt)

        ret = {
            'sdf': sdf, 'z_q': code, 'idx': codeix, 'path': path,
            'img': img, 'img_path': img_path, 'mask': mask, 'mask_path': mask_path,
            'cat_str': category, 'calibration_matrix': calibration_matrix,
            'dist_coefs': dist_coefs, 'bbox': bbox, 'smpl': self.get_smpl(data['smpl_path'], rt),
            'obj_path': mesh_obj_path
        }
        

        behave_verts, faces_idx = load_ply(data['body_mesh'])
        behave_verts = behave_verts.reshape(-1, 3).numpy()
        behave_verts = bcu.global2local(behave_verts, rt[0], rt[1])
        
        behave_verts[:, :2] *= -1
        # print(behave_verts.shape, faces_idx.shape)
        # theMesh = Meshes(verts=[torch.from_numpy(behave_verts).float()], faces=faces_idx)
        ret['body_mesh_verts'] = torch.from_numpy(behave_verts)
        ret['body_mesh_faces'] = faces_idx.reshape(-1, faces_idx.shape[-1])
        
        
        day_key = data['img_path'][len('/data/xiwang/behave/sequences/'):]

        """
        try:
            pare_verts = torch.from_numpy(self.pare[day_key]['smpl_vertices']).reshape(-1, 3)
            pare_camera = torch.from_numpy(self.pare[day_key]['orig_cam'][0])
        except:
            pare_verts =  None
            pare_camera = None 
        # print(pare_verts.shape, pare_camera.shape)
        
        ret['pare_verts'] = pare_verts
        ret['pare_camera'] = pare_camera
        """

        day_split = day_key.split('/')
        self.pare = pickle.load(open(f"/mnt/scratch/kexshi/SMPLX_Res/{day_split[0]}-{day_split[1]}-{day_split[2][0:2]}.mocap.pkl", 'rb'))
        print(self.pare.keys())
        human_pose = transforms.matrix_to_axis_angle(self.pare['body_pose'])
        human_betas = self.pare['betas']
        human_orient = torch.reshape(transforms.matrix_to_axis_angle(self.pare['global_orient']), (1,3))
        human_transl = self.pare['transl']
        print('------------------> ', human_pose.shape)
        print('------------------> ', self.pare['global_orient'].shape)
        print('------------------> ', human_orient.shape)
        ret['human_pose'] = human_pose
        ret['human_betas'] = human_betas
        ret['human_orient'] = human_orient
        ret['human_transl'] = human_transl

        #for key in self.pare[day_key].keys():
        #    print(self.pare[day_key][key].shape)

        smplx_output = self.smplx_model(return_verts=True, body_pose=torch.tensor(human_pose[None, ...]).to(self.device),
                                        betas=torch.tensor(human_betas[None, ...]).to(self.device))
        vertices = smplx_output.vertices.detach().cpu().numpy().squeeze()
        joints = smplx_output.joints.detach().cpu().numpy().squeeze()
        pelvis_transform = create_mat([0, 0, 0], joints[0], rot_type='rot_vec') \
                            @ create_mat([0, np.pi, 0], np.array([0, 0, 0]), rot_type='xyz')
        vertices = trans_pcd(vertices, np.linalg.inv(pelvis_transform))
        faces = self.smplx_model.faces
        occ_human = voxelize_mesh(vertices, faces, self.grid)
        ret['occ_human'] = occ_human[None, ...].to(torch.float32)
                    
        return ret
    
    def get_smpl(self, smpl_path, rt):
        smpl = get_smplh([smpl_path,], 'female', 'cpu')
        _, jtr, _, _ = smpl() # Compute joint 3d coordinates (1, J, 3)
        jtr = torch.cat((jtr[:, :22], jtr[:, 25:26], jtr[:, 40:41]), dim=1) # (1, 24, 3)
        jtr = jtr[0].detach().numpy() # (24, 3)
        jtr = bcu.global2local(jtr, rt[0], rt[1]) # (24, 3)
        
        return jtr
        

    def __len__(self):
        return self.N

    def name(self):
        return 'BehaveImageDataset'
    

    def create_smplx_model(self):
        self.smplx_model = smplx.create("/data/aruzzi/chairs/data/body_models/", model_type='smplx',
                                   gender='male', ext='npz',
                                   num_betas=16,
                                   use_pca=False,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=1)
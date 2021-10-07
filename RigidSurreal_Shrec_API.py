import numpy as np
import torch
import open3d as o3d
from easydict import EasyDict
import json
import h5py
import pickle
from scipy.spatial.transform import Rotation
import time
"""
For rigid surreal:
    same h5 file as given by google drive link

For SHREC:
    mkdir ./data_pkl
    tar xvf data_pkl.tar -C ./data_pkl
    at ./data_pkl_202007220823/SHREC_430sample_dict.txt please rm the prefix:
        rm ./0a0_exp3_comparing_dgfmnet/data_pkl_202007220823 for all lines 
"""

#non-rigid test only
class SHREC_pkl(torch.utils.data.Dataset):
    def __init__(self, dir_ = '../../../dataset_corr_all/data_pkl_202007220823/data_pkl_202007220823'):
        self.ratio_list=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.sample = None
        self.pclabel = None

        self.pkl_DIR = dir_
        with open('{}/SHREC_430sample_dict.txt'.format(self.pkl_DIR), 'r') as fp:
            sample_list = json.load(fp)
            self.sample_list = sample_list
            self.len = len(sample_list) #430 pairs

    def __getitem__(self,idx):
        batch_num = int(idx/10)
        in_batch_idx = idx%10 #0-0 9-9 10-0 19-9
        val_l = self.sample_list.get(str(idx))
        pkl_dir = self.pkl_DIR+val_l[0]

        pclabel_pkl_Dir = self.pkl_DIR+val_l[5]
        with open(pkl_dir, 'rb') as f:
            self.sample = pickle.load(f)
        with open(pclabel_pkl_Dir, 'rb') as f1:
            self.pclabel = pickle.load(f1)

        self.pc1 = self.sample['pc1'][in_batch_idx]
        self.pc2 = self.sample['pc2'][in_batch_idx]

        self.p_raw = self.sample['p_raw'][in_batch_idx]
        

        return_dict = {
            'src_flat':self.pc1,
            'tgt_flat':self.pc2,
            'label_flat':self.pclabel['label_0_00'], #1024 1024
        }

        for each_ratio in self.ratio_list:
            str_each_ratio = ('%.2f'%each_ratio)
            key_name_ = 'label_'+str_each_ratio.replace('.', '_')
            key_name = 'sl_'+str(each_ratio).replace('.', '')
            return_dict[key_name] = self.pclabel[key_name_]
        
        return return_dict 
 
    def __len__(self):
        return self.len
        

def UnsDeepCorr_eval_SHREC_API():
    test_dataset = SHREC_pkl()
    return test_dataset

#rigid test only, compared with DFMNET
class rigid_SURREAL_for_Ours_full_permute(torch.utils.data.Dataset): #23w//2 pairs without replace , and without template (11w5k pairs)
    def __init__(self, args, file_name, transform=None, soft_label=False, show=False, pick_out=None, train=None, npoints=None,\
        ratio_list=[0.02, 0.04, 0.06, 0.08, 0.10], gaussian_noise=False,  partition='train',factor=4):
        self.args = args
        self.gaussian_noise = gaussian_noise
        self.partition = partition
        self.factor = factor
        self.ratio_list=ratio_list
        self.file_name = file_name
        self.xyz2 = None
        with h5py.File(self.file_name, 'r') as file:
            self.len = len(file["xyz2"])
        self.pair_len = self.len//2
        self.L = list(range(0, self.len))
        self.epoch = None


    def permuted_transfrom(self, xyz1, index, random_fix=True): #N1x3
        assert(len(xyz1.shape)==2)
        assert(xyz1.shape[1]==3)
        npoint=xyz1.shape[0]
        I=np.eye(npoint) #N2xN1
        p=I.copy()
        while(np.array_equal(p,I)):
            if random_fix==True:
                np.random.seed(index)
                np.random.shuffle(p) #N2xN1 
        
        permuted_xyz1 = np.dot(p,xyz1) #N2xN1 N1x3 = N2x3
        label = p #N1xN2
        return label, permuted_xyz1

    def __getitem__(self, item):
        if (self.xyz2==None):
            self.xyz2 = h5py.File(self.file_name,'r')["xyz2"]

        pointcloud=np.array(self.xyz2[item])             #<N=1024, F=3>

        src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, corr_matrix_label=\
            self.rigid_transform_to_make_pair(pointcloud, item)
        corr_matrix_label=torch.from_numpy(corr_matrix_label).float()

        return corr_matrix_label, src, target, item

    def rigid_transform_to_make_pair(self, pointcloud, item):#input=<N, F=3>
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz) #R_ab=<3,3>
        R_ba = R_ab.T             #R_ba=<3,3>
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)]) #array([ 0.28535858,  0.27997581, -0.22740739])
        translation_ba = -R_ba.dot(translation_ab)                #array([-0.43694427, -0.10139442,  0.10163157])

        pointcloud1 = pointcloud.T #pointcloud1=<3,1024> pointcloud=<1024,3>
        

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex]) 
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1) #pointcloud2=<3,1024>
        

        euler_ab = np.asarray([anglez, angley, anglex]) 
        euler_ba = -euler_ab[::-1]                      

        if self.partition != 'train':
            np.random.seed(item)
            pointcloud1 = np.random.permutation(pointcloud1.T) #pointcloud1=<1024,3>
            np.random.seed(item)
            pointcloud2 = np.random.permutation(pointcloud2.T).T #pointcloud2=<3,1024>
            corr_matrix_label, permuted_pointcloud = self.permuted_transfrom(pointcloud1, item) #corr_matrix_label=<N1=1024,N2=1024> permuted_pointcloud=<1024,3>
            pointcloud1 = permuted_pointcloud.T #pointcloud1=<3,1024>
        elif self.partition == 'train':
            now=int(time.time())
            np.random.seed(now)
            pointcloud1 = np.random.permutation(pointcloud1.T) #pointcloud1=<1024,3>
            np.random.seed(now)
            pointcloud2 = np.random.permutation(pointcloud2.T).T #pointcloud2=<3,1024>
            corr_matrix_label, permuted_pointcloud = self.permuted_transfrom(pointcloud1, now) #corr_matrix_label=<N1=1024,N2=1024> permuted_pointcloud=<1024,3>
            pointcloud1 = permuted_pointcloud.T #pointcloud1=<3,1024>

       
        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32'), corr_matrix_label
   
    def __len__(self):
        return self.len

def get_rigid_surreal_datasets_Ours(args):
   
    train_dataset = rigid_SURREAL_for_Ours_full_permute(args, '../../../dataset_corr_all/data_training_230000_1024_3.h5',\
        partition='train',gaussian_noise=args.gaussian_noise,factor=args.factor)
  
    test_dataset = rigid_SURREAL_for_Ours_full_permute(args, '../../../dataset_corr_all/data_test_200_1024_3.h5',\
        partition='test',gaussian_noise=args.gaussian_noise,factor=args.factor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers), drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchSize,
                                            shuffle=False, num_workers=int(args.workers))
    return train_dataset, test_dataset, train_loader, test_loader

args = EasyDict({
    'batchSize':1,
    'gaussian_noise':False,
    'factor':4,
    'workers':16,
})

#surreal rigid case API
train_dataset, test_dataset, train_loader, test_loader = get_rigid_surreal_datasets_Ours(args)
for iter_, data in enumerate(train_loader):
    corr_matrix_label, src, target, item = data
    print(src.shape)
    print(target.shape)
    print(corr_matrix_label.shape)
    print(item)
    break
for iter_, data in enumerate(test_loader):
    corr_matrix_label, src, target, item = data
    print(src.shape)
    print(target.shape)
    print(corr_matrix_label.shape)
    print(item)
    break

#shrec test API (non-rigid only)
test_dataset1 = UnsDeepCorr_eval_SHREC_API()
print(test_dataset1[0].keys())
print(test_dataset1[0]['src_flat'].shape)
print(test_dataset1[0]['tgt_flat'].shape)
print(test_dataset1[0]['label_flat'].shape)

import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, Dataset
import h5py
import tables as pytables
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import os.path as osp
import glob
import time

#MY train dataset
class SURREAL_pair_without_replace_full_permute(Dataset): #23w//2 pairs without replace , and without template (11w5k pairs)
    def __init__(self, file_name, transform=None, soft_label=False, show=False, pick_out=None, train=None, npoints=None,\
        ratio_list=[0.02, 0.04, 0.06, 0.08, 0.10], partition='train'):
        self.partition=partition
        self.ratio_list=ratio_list
   
        self.file_name = file_name
        self.transform = transform
        self.soft_label = soft_label
        self.xyz2 = None
        with h5py.File(self.file_name, 'r') as file:
            self.len = len(file["xyz2"])
        self.pair_len = self.len//2
        self.L = list(range(0, self.len))
        self.epoch = None

    def get_index_pairs(self, index):
        random.seed(index)
        t1 = [self.L.pop(random.randrange(len(self.L))) for _ in range(2)]
        assert(t1[0]!=t1[1])
        return t1

    def update_epoch(self, epoch):
        if self.epoch != epoch:
            print('update epoch and L')
            self.epoch = epoch
            self.update_L_at_new_epoch()
        elif self.epoch == epoch:
            print('same epoch and keep poping L')

    def update_L(self):
        if len(self.L)==0:
            self.L = list(range(0, self.len))

    def update_L_at_new_epoch(self):
        self.L = list(range(0, self.len))

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

    def full_permute(self,pointcloud1, pointcloud2, item): #input two np point cloud
        assert(pointcloud1.shape[1]==3)
        assert(pointcloud2.shape[1]==3)
        #unordered permutation
        if self.partition != 'train':
            np.random.seed(item)
            pointcloud1 = np.random.permutation(pointcloud1) #pointcloud1=<1024,3>
            np.random.seed(item)
            pointcloud2 = np.random.permutation(pointcloud2) #pointcloud2=<1024,3>
            corr_matrix_label, permuted_pointcloud = self.permuted_transfrom(pointcloud1, item) #corr_matrix_label=<N1=1024,N2=1024> permuted_pointcloud=<1024,3>
        elif self.partition == 'train':
            now=int(time.time())
            np.random.seed(now)
            pointcloud1 = np.random.permutation(pointcloud1) #pointcloud1=<1024,3>
            np.random.seed(now)
            pointcloud2 = np.random.permutation(pointcloud2) #pointcloud2=<1024,3>
            corr_matrix_label, permuted_pointcloud = self.permuted_transfrom(pointcloud1, now) #corr_matrix_label=<N1=1024,N2=1024> permuted_pointcloud=<1024,3>
        
        return corr_matrix_label, permuted_pointcloud, pointcloud2

    def __getitem__(self, which_pair): #permuted for input1
        if (self.xyz2==None):
            # print("here once")
            self.xyz2 = h5py.File(self.file_name,'r')["xyz2"]

        self.update_L()
        t = self.get_index_pairs(which_pair)
        index1 = t[0]
        index2 = t[1]
        assert(index1!=index2)

        input1_ori=np.array(self.xyz2[index1])
        input2_ori=np.array(self.xyz2[index2])

        corr_matrix_label, permuted_input1, input2= self.full_permute(input1_ori, input2_ori, which_pair)

        input1=torch.from_numpy(permuted_input1).float()
        corr_matrix_label=torch.from_numpy(corr_matrix_label).float()
        input2=torch.from_numpy(input2).float()

        if self.transform:
            input1_ = input1.clone()
            input2_ = input2.clone()
            input1 = self.transform(input1_)        
            input2 = self.transform(input2_)
        
        if self.soft_label:
            s_label_list=[]
            # for each_ratio in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
            for each_ratio in self.ratio_list:
                s_label_list.append(self.make_soft_label(corr_matrix_label, input2, ratio=each_ratio))

            return corr_matrix_label, input1, input2, which_pair, s_label_list

        return corr_matrix_label, input1, input2, which_pair
    
    def square_distance(self, src, dst):
        N, _ = src.shape
        M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(1,0))     
        dist += torch.sum(src ** 2, -1).view(N, 1)
        dist += torch.sum(dst ** 2, -1).view(1, M)
        return dist

    def make_soft_label(self, label_origin, xyz2, ratio=0.5):
        if ratio==0.0:
            return label_origin
        else:
            label = label_origin.clone() 

            dist = self.square_distance(xyz2, xyz2) 

            max_square_radius = torch.max(dist)

            radius = ratio*torch.sqrt(max_square_radius)
    
            for row in range(label.shape[0]):   
                idx=torch.nonzero(label[row])
                
                dist_row = dist[idx.squeeze()]
                add_idx = (dist_row <= radius**2).nonzero().squeeze()
                
                if add_idx.ndimension()==0:
                    add_idx=[add_idx]
                
                for i in add_idx:
                    label[row][i]=1.0

            soft_label = label
            return soft_label

    def label_ACC_percentage(self, label_in, label_gt):
        assert(label_in.shape==label_gt.shape)
        bsize = label_in.shape[0]
        b_acc=[]
        for i in range(bsize):
            element_product = torch.mul(label_in[i], label_gt[i])     
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1)
            hit = (sum_row != 0).sum()
            acc = hit.float()/torch.tensor(N1).float()
            b_acc.append(acc*100.0)
        mean = torch.mean(torch.stack(b_acc))  
        return mean

    def label_ACC_percentage_for_inference(self, label_in, label_gt, pinput1, input2, name, sample_n_to_visualize=None):
        assert(label_in.shape==label_gt.shape)
        bsize = label_in.shape[0]
        b_acc=[]
        for i in range(bsize):
            
            element_product = torch.mul(label_in[i], label_gt[i])  
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1) #N1x1


            hit = (sum_row != 0).sum()
            acc = hit.float()/torch.tensor(N1).float()
            b_acc.append(acc*100.0)
        mean = torch.mean(torch.stack(b_acc))  
        return mean

    def corr_to_list(self, corr_matrix):
        ''' Input:    correspondence matrix c (N1xN2,{0,1}-value)
            Output:   correspondence list N1->N2= […,(i,j),…]'''
        single_corr_list = []
        try:
            pair = torch.nonzero(corr_matrix.clone().detach(), as_tuple=False) #N1xN2
        except:
            pair = torch.nonzero(corr_matrix.clone().detach()) #N1xN2
        for each in pair.tolist():
            single_corr_list.append(tuple(each))
        return single_corr_list

    def get_corr_list(self, index, mode):
        if mode=='permuted':
            corr_matrix_label, permuted_input1, input2, _ = self.__getitem__(index)
            single_corr_list = self.corr_to_list(corr_matrix_label)
            return single_corr_list, permuted_input1, input2

    def __len__(self):
        return self.pair_len

    def make_test_h5_with_label(self):
        self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.soft_label = False 

        self.partition='test'

        DESC_DICT={
            'src_flat':pytables.Float32Col(1024*3),
            'tgt_flat':pytables.Float32Col(1024*3),
            'label_flat':pytables.Float32Col(1024*1024)
        }
        for each_ratio in self.ratio_list: #soft labels
            key_name = 'sl_'+str(each_ratio).replace('.', '')
            DESC_DICT.__setitem__(key_name, pytables.Float32Col(1024*1024))

        with pytables.open_file('testset_with_soft_for_SURREAL_pair_without_replace_full_permute.h5', mode="w") as h5file:
            table = h5file.create_table('/', 'data', DESC_DICT)

            for idx in tqdm(range(self.__len__())):
                corr_matrix_label, input1, input2, which_pair = self.__getitem__(idx)

                src_flat=input1.view(1,-1).numpy()
                tgt_flat=input2.view(1,-1).numpy()
                label_flat=corr_matrix_label.view(1,-1).numpy()

                table.row['src_flat'] = src_flat
                table.row['tgt_flat'] = tgt_flat
                table.row['label_flat'] = label_flat

                for each_ratio in tqdm(self.ratio_list):
                    key_name = 'sl_'+str(each_ratio).replace('.', '')
                    val = self.make_soft_label(corr_matrix_label, input2, ratio=each_ratio) #torch.Size([1024, 1024])
                    val_flat = val.view(1,-1).numpy() #[1,1024x1024]
                    table.row[key_name]=val_flat
                table.row.append()
                table.flush()

#MY test dataset
class testset_pytable_with_soft_label(Dataset):
    def __init__(self,test_h5file_name, show=False, outname=None):
    
        self.outname=outname
        self.ratio_list = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
        self.test_h5file_name=test_h5file_name
        
        self.return_dict={}
        if isinstance(test_h5file_name, list) and len(test_h5file_name)==2:
            self.data = None
            print('two pytable files for dcp inference')
            with pytables.open_file(test_h5file_name[0], mode="r") as h5file1:
                with pytables.open_file(test_h5file_name[1], mode="r") as h5file2:
                    self.len1 = len(h5file1.get_node('/data'))
                    self.len2 = len(h5file2.get_node('/data'))           
                    self.len = self.len1+self.len2        
        else:
            self.data = None
            with pytables.open_file(test_h5file_name, mode="r") as h5file:
                self.len = len(h5file.get_node('/data'))        

    def __getitem__(self,idx):
        if isinstance(self.test_h5file_name, list) and len(self.test_h5file_name)==2:
            print('two pytable files for dcp inference')
            if idx in range(0, self.len1):
                if (self.data==None):
                    self.data=pytables.open_file(self.test_h5file_name[0], mode="r").get_node('/data')
                data_dict = self.digest_data(idx)
            elif idx in range(self.len1, self.len1+self.len2):
                if (self.data==None):
                    self.data=pytables.open_file(self.test_h5file_name[1], mode="r").get_node('/data')
                data_dict = self.digest_data(idx-self.len2)

            try:
                return {
                'src_flat':data_dict['src_flat'], 
                'tgt_flat':data_dict['tgt_flat'], 
                'label_flat':data_dict['label_flat'], 
                'scores_flat':data_dict['scores_flat'], 
                'sl_002':data_dict['sl_002'], 
                'sl_004':data_dict['sl_004'], 
                'sl_006':data_dict['sl_006'], 
                'sl_008':data_dict['sl_008'], 
                'sl_01':data_dict['sl_01'], 
                'sl_012':data_dict['sl_012'], 
                'sl_014':data_dict['sl_014'], 
                'sl_016':data_dict['sl_016'], 
                'sl_018':data_dict['sl_018'], 
                'sl_02': data_dict['sl_02'],               
                            }
            except:
                return {
                'src_flat':data_dict['src_flat'], 
                'tgt_flat':data_dict['tgt_flat'], 
                'label_flat':data_dict['label_flat'], 
                'sl_002':data_dict['sl_002'], 
                'sl_004':data_dict['sl_004'], 
                'sl_006':data_dict['sl_006'], 
                'sl_008':data_dict['sl_008'], 
                'sl_01':data_dict['sl_01'], 
                'sl_012':data_dict['sl_012'], 
                'sl_014':data_dict['sl_014'], 
                'sl_016':data_dict['sl_016'], 
                'sl_018':data_dict['sl_018'], 
                'sl_02': data_dict['sl_02'],               
                            }

        else:
            if (self.data==None):
                self.data=pytables.open_file(self.test_h5file_name, mode="r").get_node('/data')
            data_dict = self.digest_data(idx)

            try:
                return {
                'src_flat':data_dict['src_flat'], 
                'tgt_flat':data_dict['tgt_flat'], 
                'label_flat':data_dict['label_flat'], 
                'scores_flat':data_dict['scores_flat'], 
                'sl_002':data_dict['sl_002'], 
                'sl_004':data_dict['sl_004'], 
                'sl_006':data_dict['sl_006'], 
                'sl_008':data_dict['sl_008'], 
                'sl_01':data_dict['sl_01'], 
                'sl_012':data_dict['sl_012'], 
                'sl_014':data_dict['sl_014'], 
                'sl_016':data_dict['sl_016'], 
                'sl_018':data_dict['sl_018'], 
                'sl_02': data_dict['sl_02'],               
                            }
            except:

                return {
                'src_flat':data_dict['src_flat'], 
                'tgt_flat':data_dict['tgt_flat'], 
                'label_flat':data_dict['label_flat'], 

                'sl_002':data_dict['sl_002'], 
                'sl_004':data_dict['sl_004'], 
                'sl_006':data_dict['sl_006'], 
                'sl_008':data_dict['sl_008'], 
                'sl_01':data_dict['sl_01'], 
                'sl_012':data_dict['sl_012'], 
                'sl_014':data_dict['sl_014'], 
                'sl_016':data_dict['sl_016'], 
                'sl_018':data_dict['sl_018'], 
                'sl_02': data_dict['sl_02'],               
                            }

    def __len__(self):
        return self.len

    def digest_data(self,idx): 
        assert(self.data!=None)
        key_set=['src_flat', 'tgt_flat', 'label_flat', 'scores_flat'] #diff for each
        for each_ratio in self.ratio_list:
            key_name = 'sl_'+str(each_ratio).replace('.', '')
            key_set.append(key_name)
        for name_ in key_set:

            if name_== 'src_flat' or name_== 'tgt_flat':
                self.return_dict[name_] = self.data[idx][name_].reshape([1024,3])
            else: #label, scores, soft label 0.01 0.02...0.20
                if name_=='scores_flat':
                    try:
                        self.return_dict[name_] = self.data[idx][name_].reshape([1024,1024])
                    except:
                        continue
                self.return_dict[name_] = self.data[idx][name_].reshape([1024,1024])
        return self.return_dict
    

    def corr_to_list(self, corr_matrix):
        ''' Input:    correspondence matrix c (N1xN2,{0,1}-value)
            Output:   correspondence list N1->N2= […,(i,j),…]'''
        single_corr_list = []
        try:
            pair = torch.nonzero(corr_matrix.clone().detach(), as_tuple=False) #N1xN2
        except:
            pair = torch.nonzero(corr_matrix.clone().detach()) #N1xN2
        for each in pair.tolist():
            single_corr_list.append(tuple(each))
        return single_corr_list

    def label_ACC_percentage_for_inference(self, label_in, label_gt, pinput1, input2, name, sample_n_to_visualize=None):
        assert(label_in.shape==label_gt.shape)
        bsize = label_in.shape[0]
        b_acc=[]
        
        for i in range(bsize):
            
            element_product = torch.mul(label_in[i], label_gt[i])  
            N1 = label_in[i].shape[0]
            sum_row = torch.sum(element_product, dim=-1) #N1x1

            '''inference visualize using open3d'''
            idx=torch.nonzero(sum_row).squeeze()
            if idx.ndimension()==0: #make idx always ndimension==1
                idx=[idx]
            p2p_corr_matrix = label_in[i].clone()
            single_all_corr_list = self.corr_to_list(p2p_corr_matrix)
            name_out = name + '_' + str(i)+'_out.png'
 

            hit = (sum_row != 0).sum()
            acc = hit.float()/torch.tensor(N1).float()
            b_acc.append(acc*100.0)
        mean = torch.mean(torch.stack(b_acc))  
        return mean

class HumanDataModule(LightningDataModule):
    name = 'human'

    def __init__(
        self,
        data_dir: str,
        test_data_dir: str,
        val_split: float = 0.2,
        # test_split: float = 0.1,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.test_data_dir = test_data_dir if test_data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        surreal_dataset = SURREAL_pair_without_replace_full_permute(
            partition='train', 
            file_name=self.data_dir, 
            train=True, 
            npoints=1024, 
            pick_out=230000, 
            transform=None)
        test_dataset_3dcoded = testset_pytable_with_soft_label(
            test_h5file_name=self.test_data_dir,  
            outname='nonrigid_surreal',
            show=False)
        val_len = round(val_split * len(surreal_dataset)) 
        train_len = len(surreal_dataset) - val_len        
        test_len = len(test_dataset_3dcoded) 

        self.trainset, self.valset = random_split(
            surreal_dataset, 
            lengths=[train_len, val_len], 
            generator=torch.Generator().manual_seed(self.seed))
        self.testset = test_dataset_3dcoded

    def train_dataloader(self):
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader 

    def test_dataloader(self):
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

if __name__ == '__main__':
    print('ok')





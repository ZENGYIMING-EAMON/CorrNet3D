# CorrNet3D

Official implementation of our work as described in [CorrNet3D: Unsupervised End-to-end Learning of Dense Correspondence for 3D Point Clouds](https://arxiv.org/abs/2012.15638) (CVPR'21)

## Prerequisite Installation
The code has been tested with Python3.8, pytorch-lightning 1.1.6 and Cuda 10.2:

    conda create --name corrnet3d python=3.8
    conda activate corrnet3d
    pip install pytorch-lightning==1.1.6
    pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
    pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
    conda install torchvision torchaudio cudatoolkit=10.2 -c pytorch
    pip install h5py
    pip install tables
    pip install matplotlib

## Usage

### Pre-trained Models
Download the [pre-trained folder](https://drive.google.com/drive/folders/1YZknEowKevKifb1eTJWwLs5Z5YJeSUQH?usp=sharing) and put the folder under the corrnet3d folder. 

### Datasets
We provide the .h5 dataset, inherited from the main train/test dataset from 3d-coded. 
You can download the dataset we used in the paper:

- [train.h5](https://drive.google.com/file/d/1iC6A2nIMLNC0et_56-ndEELeBvhs7rsC/view?usp=sharing)
- [test.h5](https://drive.google.com/file/d/1E_gR-4rLKstYiBFWGw9sJ-3uGMcyAa04/view?usp=sharing)
### Train & Test
To test on the whole testing set, run:

    uncomment 'cli_main_test_()' in lit_corrnet3d_clean.py
    python lit_corrnet3d_clean.py --gpus=3 --batch_size=1 --ckpt_user=lightning_logs/version_114/checkpoints/epoch=43-step=202399.ckpt --data_dir=./trainset.h5 --test_data_dir=./testset.h5

To train the network, run:

    uncomment 'cli_main()' in lit_corrnet3d_clean.py
    python lit_corrnet3d_clean.py --gpus=3 --batch_size=20 --data_dir=./trainset.h5 --test_data_dir=./testset.h5




## Citation
Please cite this paper with the following bibtex:

    @inproceedings{zeng2020corrnet3d,
        author={Zeng, Yiming and Qian, Yue and Zhu, Zhiyu and Hou, Junhui and Yuan, Hui and He, Ying},
        title={CorrNet3D: Unsupervised End-to-end Learning of Dense Correspondence for 3D Point Clouds},
        booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
        year      = {2021}
    }

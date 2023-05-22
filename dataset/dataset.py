import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from torchvision import transforms
import pandas as pd
from glob import glob
from dataset.transforms import default_transforms, mode_transform


def read_off(file_path):
    with open(file_path, 'r') as file:
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

class ModelNet10Dataset(Dataset):
    def __init__(self,root,csv, mode ,valid =False, transforms=default_transforms()):
        super(ModelNet10Dataset).__init__()
        self.root = root
        self.csv = pd.read_csv(csv).reset_index()
        self.mode = mode
        self.transforms = transforms if not valid else default_transforms()

    def __len__(self):
        return self.csv.shape[0]
    
    def __preproc__(self,file_path):
        
        verts, faces = read_off(file_path)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud
    
    def __getitem__(self, index):
        row = self.csv.iloc[index]
        file_path = row['object_path']
        labels = torch.tensor(row['id_encode'])
        full_path = os.path.join(self.root, file_path)
        pointcloud = self.__preproc__(file_path=full_path)
        return {
            'pointcloud': pointcloud,
            'label': labels
        }
            
        

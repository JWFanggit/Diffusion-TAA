import os
import random
import h5py
import numpy as np
import cv2
import torch
from torchvision import transforms
from einops import rearrange, repeat, reduce
import glob
from PIL import Image
# import decord
# decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset,DataLoader
from einops import rearrange

class RAADataset(Dataset):
    def __init__(self,origin_h5_file,normal_h5_file,abnormal_h5_file):
        self.origin_h5_file=h5py.File(origin_h5_file,'r')
        self.normal_h5_file=h5py.File(normal_h5_file,'r')
        self.abnormal_h5_file=h5py.File(abnormal_h5_file,'r')
        self.origin_video_names=['ov_{}'.format(i) for i in range(1,1801)]
        self.normal_video_names=['nv_{}'.format(i) for i in range(1,1801)]
        self.normal_start_id=['start_id_{}'.format(i) for i in range(1,1801)]
        self.abnormal_video_names=['av_{}'.format(i) for i in range(1,1801)]
        self.abnormal_start_id=['start_id_{}'.format(i) for i in range(1,1801)]
        # self.video_path = self.read_video_path(root_path)
        # self.root_path=root_path
    def insert_converted_frames(self,original_video_batch, converted_video_batch, start_frames, normal):
        # for i in range(original_video_batch.shape[0]):
        original_video_batch[:, start_frames[0]:start_frames[0]+converted_video_batch.shape[1], :, :]=converted_video_batch[:,:,:,:]
        # converted_video_batch[i]
        if normal:
            label = [(torch.tensor([1, 0])) for _ in range(original_video_batch.shape[0])]
            tai = [(torch.tensor(-1)) for _ in range(original_video_batch.shape[0])]
        else:
            label = [(torch.tensor([0, 1])) for _ in range(original_video_batch.shape[0])]
            tai = start_frames
        original_video_batch = rearrange(original_video_batch, 'c f h w  -> f c h w')
        return original_video_batch, label, tai


    # def read_video_path(self, root_path):
    #
    #     # video_path = os.path.join(self.root_path, data_name)
    #     # video_path = glob.glob(root_path)
    #     # video_path = sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
    #     video_paths = os.listdir(root_path)
    #     video_path = [item for item in video_paths if item.isnumeric() and not item.endswith('txt')]
    #     return video_path
    def __len__(self):
        return len(self.origin_video_names)

    def __getitem__(self,index):
        origin_video_name= self.origin_video_names[index]
        normal_video_name = self.normal_video_names[index]
        abnormal_video_name=self.abnormal_video_names[index]
        normal_start_id=self.normal_start_id[index]
        normal_start_id = self.normal_h5_file['min_normal_dataset'][normal_start_id][:]
        # normal_start_id= normal_start_id.squeeze(0)
        origin_video_data=self.origin_h5_file['min_origin_dataset'][origin_video_name][:]
        # origin_video_data=origin_video_data.squeeze(0)
        origin_video_data=torch.from_numpy(origin_video_data)
        normal_video_data = self.normal_h5_file['min_normal_dataset'][normal_video_name][:]
        # normal_video_data=normal_video_data.squeeze(0)
        normal_video_data=torch.from_numpy(normal_video_data)
        abnormal_video_data = self.abnormal_h5_file['min_abnormal_dataset'][abnormal_video_name][:]
        # abnormal_video_data= abnormal_video_data.squeeze(0)
        abnormal_video_data=torch.from_numpy(abnormal_video_data)
        normal_video_data, label_nv, tai_nv = self.insert_converted_frames(origin_video_data, normal_video_data,
                                                                           normal_start_id, normal=True)

        abnormal_video_data, label_av, tai_av = self.insert_converted_frames(origin_video_data, abnormal_video_data,
                                                                             normal_start_id, normal=False)
        origin_video_data = rearrange(origin_video_data, 'c f h w  -> f c h w')

        data = {"ov": origin_video_data,
                "nv": normal_video_data,
                "av": abnormal_video_data,
                "start_id": normal_start_id,
                "tai_av": tai_av,
                }
        return data


if __name__=="__main__":
    # root_path=r"/media/ubuntu/Seagate Expansion Drive/HEVI-full"
    path1 = r"/media/ubuntu/Seagate Expansion Drive/h5/min-origin"
    path2 = r"/media/ubuntu/Seagate Expansion Drive/h5/min-normal"
    path3 = r"/media/ubuntu/Seagate Expansion Drive/h5/min-abnormal"

    dataset=RAADataset(path1,path2,path3)
    dataloader=DataLoader(dataset,batch_size=1,shuffle=False)
    for data in dataloader:
        print(data["ov"].shape)
        print(data["nv"].shape)
        print(data["av"].shape)
        print(data["start_id"])
        print(data["tai_av"])

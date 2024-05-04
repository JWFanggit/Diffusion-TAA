import os
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange, repeat, reduce
import glob
from PIL import Image
class DADA2KS(Dataset):
    def __init__(self, root_path, interval,phase,
                  data_aug=False):
        self.root_path = root_path
        self.interval = interval
        # self.transforms = transforms
        self.data_aug = data_aug
        self.fps = 30
        self.phase=phase
        self.data_list, self.labels, self.clips, self.toas ,self.texts= self.get_data_list()
    # def get_dict(self):
    #     fileIDs = []
    #     list_file=r'/media/ubuntu/My Passport/CAPDATA/T22.txt'
    #     with open(list_file, 'r',encoding='utf-8') as f:
    #         # for ids, line in enumerate(f.readlines()
    #         for ids, line in enumerate(f.readlines()):
    #             sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
    #             sample1 = sample[0].strip().split(' ')
    #             word = sample[1].replace('\xa0', ' ')
    #             word.strip()
    #             fileIDs.append(sample1[0])  # 1/002
    #     f = h5py.File(r'/home/ubuntu/lileilei/train.h7', "r+")
    #     # for keys in f.keys():
    #     #     print(keys)
    #     dic={}
    #     for i in range(len(fileIDs)):
    #         fileID=fileIDs[i]
    #         cond_vdata=f['x_train'][i][:6,]
    #         train_vdata=f['x_gt'][i][:16,]
    #         # x=[cond_vdata,train_vdata]
    #         dic[fileID]=[cond_vdata,train_vdata]
    #     return dic
    def get_data_list(self):
        if self.phase =="train":
            list_file = os.path.join(self.root_path+"/"+'T_new_.txt')
        # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, labels, clips, toas,texts= [], [], [], [],[]
        # samples_visited, visit_rows = [], []
            with open(list_file, 'r',encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
                    sample1=sample[0].strip().split(' ')
                    word = sample[1].replace('\xa0', ' ')
                    word.strip()
                    fileIDs.append(sample1[0])  # 1/002
                    labels.append(int(sample1[1]))  # 1: positive, 0: negative
                    clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
                    toas.append(int(sample1[4]))  # time-of-accident (toa)
                    texts.append(word.strip())
            return fileIDs, labels, clips, toas, texts

        if self.phase == "val":
            list_file = os.path.join(self.root_path + "/" + 'Tc.txt')
            # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
            assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
            fileIDs, labels, clips, toas, texts = [], [], [], [], []
            # samples_visited, visit_rows = [], []
            with open(list_file, 'r', encoding='utf-8') as f:
                # for ids, line in enumerate(f.readlines()):
                for ids, line in enumerate(f.readlines()):
                    # print(line)
                    sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
                    sample1 = sample[0].strip().split(' ')
                    for i in range(1,len(sample)):
                        sample[i]= sample[i].replace('\xa0', ' ')
                    textss=sample[1:len(sample)]
                    texts.append(textss)
                    # word = sample[1:-1]
                    # word.strip()
                    fileIDs.append(sample1[0])  # 1/002
                    labels.append(int(sample1[1]))  # 1: positive, 0: negative
                    clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
                    toas.append(int(sample1[4]))  # time-of-accident (toa)
                    # texts.append(word.strip())

            return fileIDs, labels, clips, toas, texts

    #
    # def get_abn_vdata(self):
    #     f = h5py.File("/media/ubuntu/My Passport/CAPDATA/train.h5", "r+")
    #     for keys in f.keys():
    #         # for id, values in enumerate(f[keys]):
    #         print(f[keys].shape)
    #



    def __len__(self):
        return len(self.data_list)

    def pross_video_data(self,video):
         video_datas=[]
         for fid in range(len(video)):
             video_data=video[fid]
             video_data=Image.open(video_data)
             video_data = video_data.resize((224, 224))
             video_data= np.asarray(video_data, np.float32)
             video_datas.append(video_data)

         # guide_image=video_datas[0]
         # guide_image = rearrange(guide_image, 'w h c -> c w h')
         video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
         video_data = rearrange(video_data, 'f w h c -> f c w h')
         return video_data

    def read_nomarl_rgbvideo(self, video_file, start, end):
        """Read video frames
        """
        # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
        # get the video data
        cond_video_data=video_file[start-16:start-10]
        cond_video_data=self.pross_video_data(cond_video_data)
        tran_video_data=video_file[start-16:start]
        tran_video_data=self.pross_video_data(tran_video_data)
        return cond_video_data,tran_video_data

    def read_abnomarl_rgbvideo(self, video_file, start, end):
        cond_video_data = video_file[end - 16: end - 10]
        cond_video_data = self.pross_video_data(cond_video_data)
        tran_video_data = video_file[ end - 16: end]
        tran_video_data = self.pross_video_data(tran_video_data)
        return cond_video_data, tran_video_data
    def gather_info(self, index):
        # accident_id = int(self.data_list[index].split('/')[0])
        accident_id =self.data_list[index]
        video_id = int(self.data_list[index].split('/')[1])
        texts=self.texts[index]
        # toa info
        # start, end = self.clips[index]
        # if self.labels[index] > 0: # positive sample
        #     self.labels[index]= 0,1
        #     assert self.toas[index] >= start and self.toas[index] <= end, "sample id: %s" % (self.data_list[index])
        #     toa = int((self.toas[index] - start) / self.interval)
        # else:
        #     self.labels[index] = 1, 0
        #     toa = int(self.toas[index])  # negative sample (toa=-1)
        # data_info = np.array([accident_id, video_id], dtype=np.int32)
        y=torch.tensor(self.labels[index], dtype=torch.float32)
        # data_info=torch.tensor(data_info)
        return texts ,y,accident_id


    def __getitem__(self, index):

        # if y==0:
        # clip start and ending
        start, end = self.clips[index]
        # read RGB video (trimmed)
        video_path = os.path.join(self.root_path,self.data_list[index]+"/"+"images")
        video_path=glob.glob(video_path+'/'+"*.[jp][pn]g")
        video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
        # cond_vdata,train_vdata= self.read_nomarl_rgbvideo(video_path, start, end)
        # dic=self.get_dict()
        texts,y,accident_id= self.gather_info(index)
        if y==0:
           cond_vdata,train_vdata= self.read_nomarl_rgbvideo(video_path, start, end)
           example = {
            "label":y,
            "cond_values": cond_vdata / 127.5 - 1.0,
            "pixel_values": train_vdata / 127.5 - 1.0,
            # "pixel_values": video_data / 255,
            "prompt_ids": texts,
        }
           return example
        if y==1:
           cond_vdata,train_vdata=self.read_abnomarl_rgbvideo(video_path, start, end)
           example = {
                "label":y,
                "cond_values": cond_vdata / 127.5 - 1.0,
                "pixel_values": train_vdata / 127.5 - 1.0,
                # "pixel_values": video_data / 255,
                "prompt_ids": texts,
            }
           return example


# class DADA2KS1(Dataset):
#     # def __init__(self, root_path, phase, interval=1, transforms={'image': None, 'salmap': None, 'fixpt': None},
#     #              use_salmap=True, use_fixation=True, data_aug=False):
#     def __init__(self, root_path, interval,phase,
#                   data_aug=False):
#         self.root_path = root_path
#         self.interval = interval
#         # self.transforms = transforms
#         self.data_aug = data_aug
#         self.fps = 30
#         self.phase=phase
#         self.data_list, self.labels, self.clips, self.toas ,self.texts= self.get_data_list()
#     # def get_dict(self):
#     #     fileIDs = []
#     #     list_file=r'/media/ubuntu/My Passport/CAPDATA/T22.txt'
#     #     with open(list_file, 'r',encoding='utf-8') as f:
#     #         # for ids, line in enumerate(f.readlines()
#     #         for ids, line in enumerate(f.readlines()):
#     #             sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
#     #             sample1 = sample[0].strip().split(' ')
#     #             word = sample[1].replace('\xa0', ' ')
#     #             word.strip()
#     #             fileIDs.append(sample1[0])  # 1/002
#     #     f = h5py.File(r'/media/ubuntu/My Passport/CAPDATA/train.h6', "r+")
#     #     # for keys in f.keys():
#     #     #     print(keys)
#     #     dic={}
#     #     for i in range(len(fileIDs)):
#     #         fileID=fileIDs[i]
#     #         cond_vdata=f['x_train'][i][:6,]
#     #         train_vdata=f['x_gt'][i][:16,]
#     #         # x=[cond_vdata,train_vdata]
#     #         dic[fileID]=[cond_vdata,train_vdata]
#     #     return dic
#     def get_data_list(self):
#         if self.phase =="train":
#             list_file = os.path.join(self.root_path+"/"+'Tmcvd.txt')
#         # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
#             assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
#             fileIDs, labels, clips, toas,texts= [], [], [], [],[]
#         # samples_visited, visit_rows = [], []
#             with open(list_file, 'r',encoding='utf-8') as f:
#                 # for ids, line in enumerate(f.readlines()):
#                 for ids, line in enumerate(f.readlines()):
#                     # print(line)
#                     sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
#                     sample1=sample[0].strip().split(' ')
#                     word = sample[1].replace('\xa0', ' ')
#                     word.strip()
#                     fileIDs.append(sample1[0])  # 1/002
#                     labels.append(int(sample1[1]))  # 1: positive, 0: negative
#                     clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
#                     toas.append(int(sample1[4]))  # time-of-accident (toa)
#                     texts.append(word.strip())
#             return fileIDs, labels, clips, toas, texts
#
#         if self.phase == "val":
#             list_file = os.path.join(self.root_path + "/" + 'Tc.txt')
#             # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
#             assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
#             fileIDs, labels, clips, toas, texts = [], [], [], [], []
#             # samples_visited, visit_rows = [], []
#             with open(list_file, 'r', encoding='utf-8') as f:
#                 # for ids, line in enumerate(f.readlines()):
#                 for ids, line in enumerate(f.readlines()):
#                     # print(line)
#                     sample = line.strip().split(',')  # e.g.: 1/002 1 0 149 136
#                     sample1 = sample[0].strip().split(' ')
#                     for i in range(1,len(sample)):
#                         sample[i]= sample[i].replace('\xa0', ' ')
#                     textss=sample[1:len(sample)]
#                     texts.append(textss)
#                     # word = sample[1:-1]
#                     # word.strip()
#                     fileIDs.append(sample1[0])  # 1/002
#                     labels.append(int(sample1[1]))  # 1: positive, 0: negative
#                     clips.append([int(sample1[2]), int(sample1[3])])  # [start frame, end frame]
#                     toas.append(int(sample1[4]))  # time-of-accident (toa)
#                     # texts.append(word.strip())
#
#             return fileIDs, labels, clips, toas, texts
#
#     #
#     # def get_abn_vdata(self):
#     #     f = h5py.File("/media/ubuntu/My Passport/CAPDATA/train.h5", "r+")
#     #     for keys in f.keys():
#     #         # for id, values in enumerate(f[keys]):
#     #         print(f[keys].shape)
#     #
#
#
#
#     def __len__(self):
#         return len(self.data_list)
#
#     def pross_video_data(self,video):
#          video_datas=[]
#          for fid in range(len(video)):
#              video_data=video[fid]
#              video_data=Image.open(video_data)
#              video_data = video_data.resize((128, 128))
#              video_data= np.asarray(video_data, np.float32)
#              video_datas.append(video_data)
#
#          # guide_image=video_datas[0]
#          # guide_image = rearrange(guide_image, 'w h c -> c w h')
#          video_data = np.array(video_datas, dtype=np.float32)  # 4D tensor
#          video_data = rearrange(video_data, 'f w h c -> f c w h')
#          return video_data
#
#     def read_nomarl_rgbvideo(self, video_file, start, end):
#         """Read video frames
#         """
#         # assert os.path.exists(video_file), "Path does not exist: %s" % (video_file)
#         # get the video data
#         cond_video_data=video_file[start-6:start:3]
#         cond_video_data=self.pross_video_data(cond_video_data)
#         tran_video_data=video_file[end-8:end+7:3]
#         tran_video_data=self.pross_video_data(tran_video_data)
#         return cond_video_data,tran_video_data
#
#     def gather_info(self, index):
#         # accident_id = int(self.data_list[index].split('/')[0])
#         accident_id =self.data_list[index]
#         video_id = int(self.data_list[index].split('/')[1])
#         texts=self.texts[index]
#         # toa info
#         # start, end = self.clips[index]
#         # if self.labels[index] > 0: # positive sample
#         #     self.labels[index]= 0,1
#         #     assert self.toas[index] >= start and self.toas[index] <= end, "sample id: %s" % (self.data_list[index])
#         #     toa = int((self.toas[index] - start) / self.interval)
#         # else:
#         #     self.labels[index] = 1, 0
#         #     toa = int(self.toas[index])  # negative sample (toa=-1)
#         # data_info = np.array([accident_id, video_id], dtype=np.int32)
#         y=torch.tensor(self.labels[index], dtype=torch.float32)
#         # data_info=torch.tensor(data_info)
#         return texts ,y,accident_id
#
#
#     def __getitem__(self, index):
#
#         # if y==0:
#         # clip start and ending
#         start, end = self.clips[index]
#         # read RGB video (trimmed)
#         video_path = os.path.join(self.root_path,self.data_list[index]+"/"+"images")
#         video_path=glob.glob(video_path+'/'+"*.[jp][pn]g")
#         video_path= sorted(video_path, key=lambda x: int((os.path.basename(x).split('.')[0]).split('_')[-1]))
#         # # cond_vdata,train_vdata= self.read_nomarl_rgbvideo(video_path, start, end)
#         # dic=self.get_dict()
#         # texts,y,accident_id= self.gather_info(index)
#         # if y==0:
#         cond_vdata,train_vdata= self.read_nomarl_rgbvideo(video_path, start, end)
#         return cond_vdata,train_vdata
#         # if y==1:
#         #    cond_vdata,train_vdata=dic[accident_id]
#         #    example = {
#         #         "label":y,
#         #         "cond_values": cond_vdata,
#         #         "pixel_values": train_vdata,
#         #         # "pixel_values": video_data / 255,
#         #         "prompt_ids": texts,
#         #     }
#         #    return example
if __name__=="__main__":
    train_dataset = DADA2KS(root_path=r"/media/ubuntu/My Passport/CAPDATA", interval=1,phase="train")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        pin_memory=True, drop_last=True)

    for id, batch in enumerate(train_dataloader):
            # print(step)
            print(batch["label"])
            print(batch["cond_values"].shape)
            print(batch["pixel_values"].shape)

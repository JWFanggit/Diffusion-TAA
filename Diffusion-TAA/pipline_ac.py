# from mydataset import BDDA
#存储为h5文件，对应按三元组存储;triple_orivideo,triple_normal_video,triple_abnormal_video
#all_label((1,0),(1,0),(0,1))
#all_tai(-1,-1,tai)
#bacth=1,按序列生成.
import torch
from Text2V.RAAPipeline import  RAA
import h5py
def extract_random_frames(video_bath,num_frames=22):
    # start_frames= random.sample(range(10,video_bath.shape[2]-22),video_bath.shape[0])
    start_frames=torch.randint(0,video_bath.shape[2]-22,(video_bath.shape[0],))
    extracted_videos=torch.stack([video_bath[i,:,start_frames[i]:start_frames[i]+num_frames,:,:] for i in range(video_bath.shape[0])])
    return extracted_videos,start_frames

def insert_converted_frames(original_video_batch,converted_video_batch,start_frames):
    for i in range(original_video_batch.shape[0]):
        original_video_batch[i,:,start_frames[i]:start_frames[i] +converted_video_batch.shape[2],:,:]=converted_video_batch[i]
        return original_video_batch


def  process_trible_data(path,latents,prompt,device,normal=True):
    # path = r'/media/ubuntu/Seagate Expansion Drive/best_model/best_model1'
    # prompt = ['ego-car hits a car', 'wwww', 'wwwww']
    # device = torch.device("cuda", 3)
    # latents=torch.randn(3,3,150,224,224).to(device)
    latents=latents.to(device)
    extracted_video_batch,start_frames=extract_random_frames(latents)
    extracted_video_batch=extracted_video_batch.to(device)
    extracted_video_batch=extracted_video_batch / 127.5-1
    convert_video_batch=RAA(path,extracted_video_batch,prompt,device)
    # video_batch=insert_converted_frames(latents,convert_video_batch,start_frames)
    video_batch=convert_video_batch.squeeze(0)
    if normal:
        # start_frames[:3]=-1
        tai=-1
        label=1,0
        label=torch.tensor(label,dtype=torch.float16)
    else:
        tai = start_frames
        label=0,1
        label = torch.tensor(label, dtype=torch.float16)
    return video_batch,start_frames,tai,label

if __name__=="__main__":
    import itertools
    pretrained_model_path=r"/media/ubuntu/Seagate Expansion Drive/best_model/best_model1"
    root_path=r"/media/ubuntu/Seagate Expansion Drive/HEVI-full"
    train_dataset = BDDA(root_path, 'training', interval=1, transform=False)
    # val_dataset = BDDA(root_path, 'training', interval=1, transform=False)
    # DataLoaders creation:
    normal_path=r"/media/ubuntu/Seagate Expansion Drive/h5/full-abnormal"
    list_file = r"/media/ubuntu/Seagate Expansion Drive/HEVI/prompt_c.txt"
    # ff=open(os.path.join(self.root_path, self.phase + '\word.txt'),encoding='utf-8')
    # assert os.path.exists(list_file), "File does not exist! %s" % (list_file)
    abnormal_texts = []
    # samples_visited, visit_rows = [], []
    with open(list_file, 'r', encoding='utf-8') as f:
        # for ids, line in enumerate(f.readlines()):
        for ids, line in enumerate(f.readlines()):
            prompt = line.replace('\xa0', ' ')
            abnormal_texts.append(prompt.strip())
    text_iterator = itertools.cycle(abnormal_texts)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        pin_memory=True, drop_last=True)
    global_step = 0
    with h5py.File(normal_path, 'w') as f:
        dataset = f.create_group('full_abnormal_dataset')
        for epoch in range(0,100):
            for step,video in enumerate(train_dataloader):
                #MIN：Raw Videos：512
                #生成：正样本1800，负样本1800
                global_step=global_step+1
                print(global_step)
                abnormal_prompt=next(text_iterator)
                abnormal_prompt=[abnormal_prompt]
                device = torch.device("cuda", 2)
                video =  video.to(device)
                abnormal_video,start_id, tai_abnormal,label= process_trible_data(pretrained_model_path, video, abnormal_prompt, device,normal=False)
                abnormal_video=abnormal_video.cpu().numpy()
                start_id=start_id.cpu().numpy()
                # tai_normal= tai_normal.numpy()
                label=label.numpy()
                dataset.create_dataset(f'av_{global_step}',data=abnormal_video)
                dataset.create_dataset(f'start_id_{global_step}', data=start_id)
                dataset.create_dataset(f'tai_abnormal_{global_step}', data=tai_abnormal,dtype=int)
                dataset.create_dataset(f'label_{global_step}', data=label)
                dataset.create_dataset(f'abnormal_prompt_{global_step}',data=abnormal_prompt,dtype='S100')
                if global_step >= 1600:
                   break
            if global_step >= 1600:
                break
        f.flush()
        f.close()

import os
import glob
rootpath = r'/media/ubuntu/Seagate Expansion Drive/HEVI'
video_paths = os.listdir(rootpath)
video_paths = [item for item in video_paths if item.isnumeric() and not item.endswith('txt')]
for video_name in video_paths:
    video_path=os.path.join(rootpath,video_name)
    videos= glob.glob(video_path + '/' + "*.jpg")

    print(len(videos),video_name)
    if len(videos)<=150:
        print("xxxxxxxxxxxxxxx")

# for i in range(0, 58):
#     print(i)
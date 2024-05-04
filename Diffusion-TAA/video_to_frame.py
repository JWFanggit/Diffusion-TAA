import numpy as np
import cv2
import time
import os
import os.path as osp
currentDirectory = os.getcwd()
# video_directory = 'Normal/'
video_directory=r'/media/ubuntu/My Passport/hevi/videos'
destination =r'/media/ubuntu/Seagate Expansion Drive/HEVI'
video = os.listdir(video_directory)
i=0
for j in video:
    path1=os.path.join(video_directory,j)
    count = 0
    i=i+1
    # vname_slice = j.split('.')
    # print(vname_slice[0])
    cap = cv2.VideoCapture(path1)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
#     print(fps)
    images =[]
#     check if capture was successful
    if not cap.isOpened():
        print("Could not open!")
    else:
        print("Video read successful!")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Extracting frames from: ',  path1)
        for loop in range(total_frames):
            cap = cv2.VideoCapture(path1)
            cap.set(1,loop)
#             cap.set(cv2.CAP_PROP_POS_MSEC,(count*55.1))
# #             cv2.CAP_PROP_POS_MSEC,(count*1000)

            success = cap.grab()
            ret, image = cap.retrieve()
            try:
                image = cv2.resize(image, (224, 224))
            except:
                continue

            frame_name ='_frame_%d.jpg' %loop
            images.append(frame_name)
            saved_path = str(i)
            destination_2 = destination + '/'+ saved_path
            if not os.path.exists(destination_2):
                os.makedirs(destination_2)
#                 print(destination_2)

            destination_dir = osp.join(destination_2, frame_name)
            cv2.imwrite(destination_dir,image)
#             if count % hop == 0:
#                 cv2.imwrite(destination_dir,image)
    count = count + 1
    cap.release()
    cv2.destroyAllWindows()
    print('Finished')
print('+++++++++++++++++++++++++++++')
print('Completed : ', count)








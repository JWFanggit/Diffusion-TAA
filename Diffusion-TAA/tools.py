import numpy as np
# import matplotlib.pyplot as plt
import os
from scipy.interpolate import make_interp_spline
import torch.nn.functional as F
import matplotlib.pyplot as plt
def evaluate_earliness(all_pred, all_labels, time_of_accidents, fps=30.0, thresh=0.5):
    """Evaluate the earliness for true positive videos"""
    time = 0.0
    counter = 0
    # iterate each video sample
    for i in range(len(all_pred)):
        pred_bins = (all_pred[i] >= thresh).astype(int)
        inds_pos = np.where(pred_bins > 0)[0]
        if all_labels[i] > 0 and len(inds_pos) > 0:
            # only true positive video needs to compute earliness
            time += max((time_of_accidents[i] - inds_pos[0]) / fps, 0)
            counter += 1  # number of TP videos
    mTTA = time / counter if counter > 0 else 0 # average TTA (seconds) per-video
    return mTTA
def evaluation(all_pred, all_labels, time_of_accidents, fps=30.0):
    """
    :param: all_pred (N x T), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """

    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if all_labels[idx] > 0:
            pred = all_pred[idx, :int(toa)]  # positive video
        else:
            pred = all_pred[idx, :]  # negative video
        # find the minimum prediction
        min_pred = np.min(pred) if min_pred > np.min(pred) else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = all_pred.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5)
    Precision = np.zeros((n_frames))
    Recall = np.zeros((n_frames))
    Time = np.zeros((n_frames))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        Tp_Tn = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*all_labels[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(all_labels) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(all_labels)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80
import cv2
import torch


def print_results(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all, result_dir):
    result_file = os.path.join(result_dir, 'eval_all.txt')
    with open(result_file, 'w') as f:
        for e, APvid, AP, mTTA, TTA_R80, Un in zip(Epochs, APvid_all, AP_all, mTTA_all, TTA_R80_all, Unc_all):
            f.writelines('Epoch: %s,'%(e) + ' APvid={:.3f}, AP={:.3f}, mTTA={:.3f}, TTA_R80={:.3f}, mAU={:.5f}, mEU={:.5f}\n'.format(APvid, AP, mTTA, TTA_R80, Un[0], Un[1]))
    f.close()


def vis_results(vis_data, toa,labels,batch_size, vis_dir, smooth=False):
    # assert vis_batchnum <= len(vis_data)
    # for b in range(vis_batchnum):
        # results = vis_data[b]
        # pred_frames = results['pred_frames']
        # labels = results['label']
        # toa = results['toa']
        # video_ids = results['video_ids']
        # detections = results['detections']
        # uncertainties = results['pred_uncertain']
        for n in range(batch_size):
            pred_frames=vis_data[n, :]
            pred_mean = vis_data[n, :]  # (90,)
            # pred_std_alea = 1.0 * np.sqrt(uncertainties[n, :, 0])
            # pred_std_epis = 1.0 * np.sqrt(uncertainties[n, :, 1])
            xvals = range(len(pred_mean))
            if smooth:
                # sampling
                xvals = np.linspace(0,len(pred_mean)-1,20)
                pred_mean_reduce = pred_mean[xvals.astype(np.int)]
                pred_std_alea_reduce = pred_std_alea[xvals.astype(np.int)]
                pred_std_epis_reduce = pred_std_epis[xvals.astype(np.int)]
                # smoothing
                xvals_new = np.linspace(1,len(pred_mean)+1,80)
                pred_mean = make_interp_spline(xvals, pred_mean_reduce)(xvals_new)
                # pred_std_alea = make_interp_spline(xvals, pred_std_alea_reduce)(xvals_new)
                # pred_std_epis = make_interp_spline(xvals, pred_std_epis_reduce)(xvals_new)
                pred_mean[pred_mean >= 1.0] = 1.0-1e-3
                xvals = xvals_new
                # fix invalid values
                indices = np.where(xvals <= toa[n])[0]
                xvals = xvals[indices]
                pred_mean = pred_mean[indices]
                pred_std_alea = pred_std_alea[indices]
                pred_std_epis = pred_std_epis[indices]
            # plot the probability predictions
            fig, ax = plt.subplots(1, figsize=(24, 3.5))
            ax.fill_between(xvals, pred_mean - pred_std_alea, pred_mean + pred_std_alea, facecolor='wheat', alpha=0.5)
            ax.fill_between(xvals, pred_mean - pred_std_epis, pred_mean + pred_std_epis, facecolor='yellow', alpha=0.5)
            plt.plot(xvals, pred_mean, linewidth=3.0)
            if toa[n] <= pred_frames.shape[1]:
                plt.axvline(x=toa[n], ymax=1.0, linewidth=3.0, color='r', linestyle='--')
            # plt.axhline(y=0.7, xmin=0, xmax=0.9, linewidth=3.0, color='g', linestyle='--')
            # draw accident region
            x = [toa[n], pred_frames.shape[1]]
            y1 = [0, 0]
            y2 = [1, 1]
            ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            fontsize = 25
            plt.ylim(0, 1.1)
            plt.xlim(1, pred_frames.shape[1])
            plt.ylabel('Probability', fontsize=fontsize)
            plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
            plt.xticks(range(0, pred_frames.shape[1], 10), fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.grid(True)
            plt.tight_layout()
            tag = 'pos' if labels[n] > 0 else 'neg'
            plt.savefig(os.path.join(vis_dir, str(n) + '_' + tag + '.png'))
            plt.close()
            # plt.show()



def read_frames_from_videos(root_path, vid_name, start, end, folder, phase='testing', interval=1):
    """Read video frames
    """
    video_path = os.path.join(root_path, phase, folder, vid_name + '.avi')
    assert os.path.exists(video_path), "Path does not exist: %s" % (video_path)
    # get the video data
    cap = cv2.VideoCapture(video_path)
    video_data = []
    for fid in range(start, end + 1, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        assert ret, "read video failed! file: %s frame: %d" % (video_path, fid)
        video_data.append(frame)
    video_data = np.array(video_data)
    return video_data


def create_curve_video(pred_scores, toa, n_frames, frame_interval):
    # background
    fig, ax = plt.subplots(1, figsize=(30, 5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames + 1)
    plt.ylabel('Probability', fontsize=fontsize)
    plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames * frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    # plt.savefig('tmp_curve.png')
    # draw curves
    from matplotlib.animation import FFMpegWriter
    curve_writer = FFMpegWriter(fps=2, metadata=dict(title='Movie Test', artist='Matplotlib', comment='Movie support!'))
    with curve_writer.saving(fig, "tmp_curve_video.mp4", 100):
        xvals = np.arange(n_frames + 1) * frame_interval
        pred_scores = pred_scores.tolist() + [pred_scores[-1]]
        for t in range(1, n_frames + 1):
            plt.plot(xvals[:(t + 1)], pred_scores[:(t + 1)], linewidth=5.0, color='r')
            plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
            if toa >= 0:
                plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
                x = [toa, n_frames * frame_interval]
                y1 = [0, 0]
                y2 = [1, 1]
                ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
            curve_writer.grab_frame()
    plt.close()
    # read frames
    cap = cv2.VideoCapture("tmp_curve_video.mp4")
    ret, frame = cap.read()
    curve_frames = []
    while (ret):
        curve_frames.append(frame)
        ret, frame = cap.read()
    return curve_frames


def plot_scores(pred_scores, toa, n_frames, frame_interval, out_file):
    # background
    fig, ax = plt.subplots(1, figsize=(30, 5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames + 1)

    xvals = np.arange(n_frames) * frame_interval
    plt.plot(xvals, pred_scores, linewidth=5.0, color='r')
    plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
    if toa >= 0:
        plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        x = [toa, n_frames * frame_interval]
        y1 = [0, 0]
        y2 = [1, 1]
        ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)

    # plt.ylabel('Probability', fontsize=fontsize)
    # plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames * frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def minmax_norm(salmap):
    """Normalize the saliency map with min-max
    salmap: (B, 1, H, W)
    """
    batch_size, height, width = salmap.size(0), salmap.size(2), salmap.size(3)
    salmap_data = salmap.view(batch_size, -1)  # (B, H*W)
    min_vals = salmap_data.min(1, keepdim=True)[0]  # (B, 1)
    max_vals = salmap_data.max(1, keepdim=True)[0]  # (B, 1)
    salmap_norm = (salmap_data - min_vals) / (max_vals - min_vals + 1e-6)
    salmap_norm = salmap_norm.view(batch_size, 1, height, width)
    return salmap_norm


def generate_attention(frame_data, data_trans, salmodel, fovealmodel, fixations, image_size, n_slice=1, rho_list=None,
                       device=None):
    assert frame_data.shape[0] % n_slice == 0, "invalid n_slice!"
    slice_size = int(frame_data.shape[0] / n_slice)
    attention_maps = []
    for i in range(n_slice):
        # get bottom-up attention
        input_data = torch.FloatTensor(data_trans(frame_data[i * slice_size:(i + 1) * slice_size])).to(device)
        fixation_data = torch.from_numpy(fixations[i * slice_size:(i + 1) * slice_size]).to(device)
        foveal_data = fovealmodel.foveate(input_data, fixation_data)
        with torch.no_grad():
            saliency_bu = salmodel(input_data)
            saliency_bu = minmax_norm(saliency_bu)
            saliency_td = salmodel(foveal_data)
            saliency_td = minmax_norm(saliency_td)
        saliency_bu = saliency_bu.squeeze(1).cpu().numpy()
        saliency_td = saliency_td.squeeze(1).cpu().numpy()
        rho = np.expand_dims(np.expand_dims(np.array(rho_list[i * slice_size:(i + 1) * slice_size]), axis=1), axis=2)
        saliency = (1 - rho) * saliency_bu + rho * saliency_td
        # padd the saliency maps to image size
        salmap = saliency_padding(saliency, image_size)
        attention_maps.append(salmap)
    attention_maps = np.concatenate(attention_maps, axis=0)
    return attention_maps


def saliency_padding(saliency, image_size):
    """Up padding the saliency (B, 60, 80) to image size (B, 330, 792)
    """
    # get size and ratios
    height, width = saliency.shape[1:]
    rows_rate = image_size[0] / height  # h ratio (5.5)
    cols_rate = image_size[1] / width  # w ratio (9.9)
    # padding
    if rows_rate > cols_rate:
        pass
    else:
        new_rows = (image_size[0] * width) // image_size[1]
        patch_ctr = saliency[:, ((height - new_rows) // 2):((height - new_rows) // 2 + new_rows), :]
        patch_ctr = np.rollaxis(patch_ctr, 0, 3)
        padded = cv2.resize(patch_ctr, (image_size[1], image_size[0]))
        padded = np.rollaxis(padded, 2, 0)
    return padded





import numpy as np
import matplotlib.pyplot as plt

def smooth_data(data, window_size):
    # 使用滑动窗口计算平均值
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

def plot_scores1(pred_scores, toa, n_frames, frame_interval, out_file, smoothing_window=5):
    # background
    fig, ax = plt.subplots(1, figsize=(30,5))
    fontsize = 25
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)

    xvals = np.arange(n_frames) * frame_interval
    smoothed_scores = smooth_data(pred_scores, smoothing_window)  # 平滑处理得分数据
    plt.plot(xvals, smoothed_scores, linewidth=5.0, color='r')
    plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
    if toa >= 0:
        plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        x = [toa, n_frames * frame_interval]
        y1 = [0, 0]
        y2 = [1, 1]
        ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)
    plt.xticks(range(0, n_frames*frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()



#
# import numpy as np
# import matplotlib.pyplot as plt

def smooth_data1(data, window_size):
    # 使用滑动窗口计算平均值
    window = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, window, mode='same')
    return smoothed_data

def plot_scores2(pred_scores, toa, n_frames, frame_interval, out_file, smoothing_window=5):
    # background
    fig, ax = plt.subplots(1, figsize=(30,5))
    fontsize = 50
    plt.ylim(0, 1.0)
    plt.xlim(0, n_frames+1)
    xvals = np.arange(n_frames) * frame_interval
    for i in range(len(pred_scores)):
        smoothed_scores = smooth_data1(pred_scores[i], smoothing_window)  # 平滑处理得分数据
        # curve=f"Curve{i}"
        if i == 0:
            plt.plot(xvals, smoothed_scores,linewidth=8.0, color='fuchsia')
        if i==1:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=7.0,color='deepskyblue' )
        if i==2:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=7.0, color='darksalmon')

        if i == 3:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=7.0, color='red')
        if i==4:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=7.0, color='darkolivegreen')
        # plt.show()
        if i == 5:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=8.0, color='yellowgreen')

        if i == 6:
            plt.plot(xvals, smoothed_scores, linestyle='dashed', linewidth=4.0, color='blue')



    plt.axhline(y=0.5, xmin=0, xmax=n_frames + 1, linewidth=3.0, color='g', linestyle='--')
    if toa >= 0:
        plt.axvline(x=toa, ymax=1.0, linewidth=3.0, color='r', linestyle='--')
        x = [toa, n_frames * frame_interval]
        y1 = [0, 0]
        y2 = [1, 1]
        ax.fill_between(x, y1, y2, color='C1', alpha=0.3, interpolate=True)

    plt.ylabel('Probability', fontsize=fontsize)
    # plt.xlabel('Frame (FPS=30)', fontsize=fontsize)
    plt.xticks(range(0, n_frames*frame_interval + 1, 10), fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # handles=[plt.Line2D([0],[0],color=f"C{i}",linewidth=5) for i in range (len((pred_scores)))]
    # plt.legend(fontsize=fontsize,loc='upper left')
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()




def calculate_variance(scores, label):
    if label == 1:
        start_frame = np.argmax(scores > 0.5)
        scores=scores[start_frame:]
        # if  start_frame >0:
        variance = np.var(scores)
        # else:
        #     variance=0
    else:
        variance = np.var(scores)
    return variance


def calculate_average_variance(batch_scores, batch_labels):
    total_variance = 0
    for scores, label in zip(batch_scores, batch_labels):
        variance = calculate_variance(scores, label)
        # print(variance)
        total_variance += variance
    # average_variance = total_variance / len(batch_scores)
    return total_variance


def caculate_uncertainty(outputs):
    # outputs = torch.randn(b, 2)  # 示例输出

    # 计算不确定性
    p = F.softmax(outputs, dim=1)  # 应用softmax函数，将输出转换为概率分布

    # 计算aleatoric不确定性
    p_diag = torch.diag_embed(p)  # 计算概率分布的对角矩阵
    p_cov = torch.matmul(p.unsqueeze(-1), p.unsqueeze(-1).transpose(1, 2))# 计算概率分布的协方差矩阵
    uncertain_alea = torch.mean(p_diag - p_cov, dim=0)  # 计算aleatoric不确定性
    uncertain_alea_= torch.abs(uncertain_alea[:,0])+ torch.abs(uncertain_alea[:,1])
    # 计算epistemic不确定性
    p_bar = torch.mean(p, dim=0)  # 计算概率分布的均值
    p_diff_var = torch.matmul((p - p_bar).unsqueeze(-1),
                              (p - p_bar).unsqueeze(-1).transpose(1, 2))  # 计算概率分布与均值之间的差的协方差矩阵
    uncertain_epis = torch.mean(p_diff_var, dim=0)  # 计算epistemic不确定性
    uncertain_epis_=torch.abs(uncertain_epis[:,0])+torch.abs(uncertain_epis[:,1])
    # uncertainly=torch.diagonal(uncertain_alea,dim1=-2,dim2=-1)+torch.diagonal(uncertain_epis,dim1=-2,dim2=-1)
    # return uncertainly,uncertain_alea, uncertain_epis
    return  uncertain_alea_, uncertain_epis_


# 示例数据
batch_size = 2
num_frames = 150
batch_scores = np.random.rand(batch_size, num_frames)
batch_labels = np.random.randint(0, 2, size=batch_size)

average_variance = calculate_average_variance(batch_scores, batch_labels)
# print(average_variance.size())
# print("Average Variance:", average_variance)


def caculate_fvd(real_recon,fake):
    reals=real_recon
    device=real_recon.device
    i3d = InceptionI3d(400, in_channels=3).to(device)
        # filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    #you can download at https://github.com/XianyuanLiu/pytorch-i3d
    filepath=r"...../rgb_imagenet.pt"
    i3d.load_state_dict(torch.load(filepath, map_location=device))
    i3d.eval()
    fake_embeddings = []
    # batch['video']=torch.randn(1,3,22,224,224)
    # for i in range(0, batch['video'].shape[0], MAX_BATCH):
    #     fake = videogpt.sample(MAX_BATCH, {k: v[i:i+MAX_BATCH] for k, v in batch.items()})
    # fake=torch.randn(2,3,22,224,224)
    fake = fake.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    fake = (fake * 255).astype('uint8')
    fake_embeddings.append(get_fvd_logits(fake, i3d=i3d, device=device))
    fake_embeddings = torch.cat(fake_embeddings)

    # real = batch['video'].to(device)
    real_recon_embeddings = []
    # for i in range(0, batch['video'].shape[0], MAX_BATCH):
    #     real_recon = (videogpt.get_reconstruction(batch['video'][i:i+MAX_BATCH]) + 0.5).clamp(0, 1)
    # real_recon=torch.randn(2,3,22,224,224)
    real_recon = real_recon.permute(0, 2, 3, 4, 1).cpu().numpy()
    real_recon = (real_recon * 255).astype('uint8')
    real_recon_embeddings.append(get_fvd_logits(real_recon, i3d=i3d, device=device))
    real_recon_embeddings = torch.cat(real_recon_embeddings)

    # aaaa=torch.randn(2,3,22,224,224)
    # real = aaaa+ 0.5
    real=reals+0.5
    real = real.permute(0, 2, 3, 4, 1).cpu().numpy() # BCTHW -> BTHWC
    real = (real * 255).astype('uint8')
    real_embeddings = get_fvd_logits(real, i3d=i3d, device=device)
    # 要求videos在（-1，1），因此可以直接使用我的数据
    # fake_embeddings = all_gather(fake_embeddings)
    # real_recon_embeddings = all_gather(real_recon_embeddings)
    # real_embeddings = all_gather(real_embeddings)

    assert fake_embeddings.shape[0] == real_recon_embeddings.shape[0] == real_embeddings.shape[0] == 3

    fvd = frechet_distance(fake_embeddings.clone(), real_embeddings)
    fvd_star = frechet_distance(fake_embeddings.clone(), real_recon_embeddings)

    print(f"FVD: {fvd.item()}, FVD*: {fvd_star.item()}")
    return fvd,fvd_star


def caculate_clip(input_data,prompt,model,preprocess):
    # 加载CLIP模型
    device =input_data.device
    # 定义文本
    text = clip.tokenize(prompt).to(device)
    # 定义输入数据
    # input_data = torch.randn(1, 3, 22, 224, 224).to(device)
    # 初始化变量
    total_score = 0
    batch_size = input_data.shape[0]
    # 循环处理每个视频
    for i in range(batch_size):
        # 获取当前视频的帧数
        num_frames = 22
        # print(num_frames)
        # 初始化变量
        video_score = 0
        # 循环处理每一帧
        for j in range(0,num_frames):
            # 将当前帧转换为模型所需的格式
            # print(j)
            image = input_data[:,:,j,:].to(device)
            # print(image.shape)
            # image = preprocess(frame).unsqueeze(0).to(device)
            # 计算相似度得分
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1,keepdim=True)
                # image_features=image_features.cpu().numpy()
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # text_features=text_features.cpu().numpy()
                score = (100.0 * image_features @ text_features.T)
                clip_score=torch.diag(score).sum()

            # 获取CLIP Score并累加到视频分数中
            # clip_score = similarity[0, 0].item()
            video_score += clip_score

        # 计算平均CLIP Score并累加到总分数中
        if num_frames > 0:
            avg_clip_score = video_score / num_frames
            total_score += avg_clip_score

    # 计算平均CLIP Score
    if batch_size > 0:
        avg_clip_score = total_score / batch_size
    else:
        avg_clip_score = 0.0

    print("avg_clip_score:",avg_clip_score)
    return avg_clip_score


def calculate_fid(x1, x2,model):
    device=x1.device
    inception =model.eval().to(device)

    mu, cov = [], []
    for loader in [x1, x2]:
        actvs = []
        for i in range(loader.shape[2]):
            x=loader[:,:,i,:,:]
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    print('FID: ', fid_value)
    return fid_value
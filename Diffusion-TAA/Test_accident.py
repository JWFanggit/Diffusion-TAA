# from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from norm import online_min_max_normalization
from vivitt import Accident

from tqdm import tqdm
import os
import numpy as np
from tools import evaluation, evaluate_earliness, plot_scores1, plot_scores2
from mydataset import CAP1
from mydataset import DADA
# from mydataset import CAP
from mydataset import ccd

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# device = ("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:1')
num_epochs = 50
# learning_rate = 0.0001
batch_size = 2
shuffle = True
pin_memory = True
num_workers = 1
rootpath1=r'.../DADA_small'
# rootpath2 = r'/home/ubuntu/lileilei/videos-frame'
# rootpath3 = r'/media/ubuntu/My Passport/CAPDATA'
frame_interval = 1
input_shape = [224, 224]
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
val_data1 = DADA(rootpath1, 'testing', interval=1, transform=transform)
# val_data=CAP(rootpath , 'testing', interval=1,transform=transform)
# val_data2 = ccd(rootpath2, 'testing', interval=1, transform=transform)
# val_data3 = CAP1(rootpath3, 'testing', interval=1, transform=transform)
valdata_loader1 = DataLoader(dataset=val_data1, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True, drop_last=True)
#
# valdata_loader2 = DataLoader(dataset=val_data2, batch_size=1, shuffle=False,
#                              num_workers=4, pin_memory=True, drop_last=True)
#
# valdata_loader3 = DataLoader(dataset=val_data3, batch_size=1, shuffle=False,
#                              num_workers=4, pin_memory=True, drop_last=True)
#
def calculate_average_variance(batch_scores, batch_labels):
    total_variance = 0
    for scores, label in zip(batch_scores, batch_labels):
        variance = calculate_variance(scores, label)
        # print(variance)
        total_variance += variance
    # average_variance = total_variance / len(batch_scores)
    return total_variance

def write_scalars(logger, epoch, loss):
    logger.add_scalars('train/loss', {'loss': loss}, epoch)


def write_test_scalars(logger, epoch, losses, metrics):
    # logger.add_scalars('test/loss',{'loss':loss}, epoch)
    logger.add_scalars('test/losses/total_loss', {'Loss': losses}, epoch)
    logger.add_scalars('test/accuracy/AP', {'AP': metrics['AP'], 'PR80': metrics['PR80']}, epoch)
    logger.add_scalars('test/accuracy/time-to-accident', {'mTTA': metrics['mTTA'], 'TTA_R80': metrics['TTA_R80']},
                       epoch)

def online_min_max_normalization(scores, threshold, start_frame):
    normalized_scores = scores.copy()
    for i in range(start_frame, len(scores)):
        if scores[i] > threshold:
            max_score = max(scores[start_frame:i+1])
            min_score = min(scores[start_frame:i+1])
            # print(min_score)
            if max_score != min_score:
                normalized_scores[i] = (scores[i] - min_score) / (max_score - min_score)
                # print(normalized_scores[i])
            else:
                normalized_scores[i]=min_score
            # normalized_scores[i] = (scores[i] - min_score) / (max_score - min_score)+1e-8
    return normalized_scores


def test(test_dataloader, model):
    all_pred = []
    all_labels = []
    all_toas = []
    model.eval()
    all_min_max_var = []
    with torch.no_grad():
        loop = tqdm(test_dataloader, total=len(test_dataloader), leave=True)
        for imgs, info, label in loop:
            # torch.cuda.empty_cache()
            imgs = imgs.to(device)
            labels = label
            toa = info[0:, 0].to(device)
            labels = np.array(labels).astype(int)
            labels = torch.from_numpy(labels)
            labels = labels.to(device)
            #You should make the model to output the accident score only,delete the loss.
            outputs = model(imgs)
            num_frames = imgs.size()[1]
            batch_size = imgs.size()[0]
            pred_frames = np.zeros((batch_size, num_frames), dtype=np.float32)
            for t in range(num_frames):
                pred = outputs[t]
                pred = pred.cpu().numpy() if pred.is_cuda else pred.detach().numpy()
                pred_frames[:, t] = np.exp(pred[:, 1]) / np.sum(np.exp(pred), axis=1)
            # gather results and ground truth
            label_onehot = labels.cpu().numpy()
            label = np.reshape(label_onehot[:, 1], [batch_size, ])
            pred_scoress = []
            for i in range(pred_frames.shape[0]):
                pred_scores = pred_frames[i, :]
                pred_scores = online_min_max_normalization(pred_scores, 0.5, 0)
                pred_scores = np.expand_dims(pred_scores, axis=0)
                pred_scoress.append(pred_scores)
            all_pred_score = np.concatenate(pred_scoress)
            min_max_var = calculate_average_variance(all_pred_score, label)
            all_pred.append(pred_frames)
            all_min_max_var.append(min_max_var)
            all_labels.append(label)
            toas = np.squeeze(toa.cpu().numpy()).astype(np.int64)
            all_toas.append(toas)
        all_pred = np.concatenate(all_pred)
        all_labels = np.concatenate(all_labels)
        all_toas = np.concatenate(all_toas)
        all_min_max_var = np.array(all_min_max_var)

        return all_pred, all_labels, all_toas,all_min_max_var

def test_data():
    validation_steps = 100,
    train_batch_size = 1,
    max_train_steps = 500,
    learning_rate = 3e-5,
    scale_lr = False,
    lr_scheduler = "constant",
    lr_warmup_steps = 0,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_weight_decay = 1e-2,
    adam_epsilon = 1e-08,
    image_size = 224,
    patch_size = 16,
    num_frames = 5,
    input_dim = 192,
    hidden_dim = 128,
    num_classes = 2,
    heads = 8,
    depth = 6,
    in_channels = 3,
    dim_head = 64,
    scale_dim = 4,
    dropout = 0.1,
    emb_dropout = 0.,
    drop_path_rate = 0.1,
    max_grad_norm = 10.0,
    pool = "mean"
    resume_from_checkpoint = None,
    mixed_precision = "fp16",
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    ckpt_path= r"..../pytorch_model.bin"
    weight = torch.load(ckpt_path, map_location='cpu')
    model = Accident(image_size=224, patch_size=16, num_classes=2, num_frames=5, dim=192, depth=3, heads=8, pool='mean',
                     in_channels=3, dim_head=64, dropout=0.,
                     emb_dropout=0., scale_dim=4, drop_path_rate=0, norm_layer=None)
    model.eval()
    model.load_state_dict(weight)
    model = model.to(device)
    print('------Starting evaluation------')
    all_pred, all_labels, all_toas,all_min_max_var, = test(valdata_loader2, model)
    mTTA = evaluate_earliness(all_pred, all_labels, all_toas, fps=30, thresh=0.5)
    print("\n[Earliness] mTTA@0.5 = %.4f seconds." % (mTTA))
    AP, mTTA, TTA_R80 = evaluation(all_pred, all_labels, all_toas, fps=30)
    print("[Correctness] AP = %.4f, mTTA = %.4f, TTA_R80 = %.4f" % (AP, mTTA, TTA_R80))
    average_min_max_var = np.mean(all_min_max_var)
    print(" average_min_max_var", average_min_max_var)



if __name__ == "__main__":
    test_data()


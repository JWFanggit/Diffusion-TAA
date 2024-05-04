import torch
import lpips

class VideoLPIPS(torch.nn.Module):
    def __init__(self, net='alex', version='0.1'):
        super(VideoLPIPS, self).__init__()
        self.lpips = lpips.LPIPS(net=net, version=version)

    def forward(self, x, y,batch_size):
        """
        x, y: (batchsize, 4, frames, 28, 28) video feature tensors
        return: video-level LPIPS loss
        """
        # reshape to (batchsize*frames, 4, 28, 28)
        device=x.device
        self.lpips=  self.lpips.to(device)
        b,c,f,h,w=x.shape
        x = x.transpose(1, 2).reshape(-1, c, h, w)
        y = y.transpose(1, 2).reshape(-1, c, h, w)

        # calculate LPIPS loss
        loss = self.lpips(x, y)
        loss=torch.clamp(loss,0,1)
        # reshape to (batchsize, frames)
        loss = loss.reshape(-1, x.shape[0] // (batch_size)).mean(dim=1)

        # calculate video-level LPIPS loss
        loss = loss.mean()

        return loss
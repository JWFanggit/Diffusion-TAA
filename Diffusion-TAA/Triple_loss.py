import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self,margin):
        super(TripletLoss,self).__init__()
        self.margin=margin

    def forward(self,anchor,positive,negative):
        pos_dist=torch.sum((anchor-positive)**2,dim=-1)
        neg_dist=torch.sum((anchor-negative)**2,dim=-1)
        basic_loss=pos_dist-neg_dist+self.margin
        loss=torch.mean(torch.max(basic_loss,torch.zeros_like(basic_loss)))
        return loss
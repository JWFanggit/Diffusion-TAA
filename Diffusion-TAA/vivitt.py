import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
from ee import SelfAttention
import numpy as np
from torch.autograd import Variable
from ats_block import ATSBlock, Block
from functools import partial
from Patch_Embedding import PatchEmbed
import random
from Triple_loss import TripletLoss


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MHAttention(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        # self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                SelfAttention(dim)
            )

    def forward(self, x):
        for attn in self.layers:
            # x = attn(x) + x
            x = attn(x)
            # x = ff(x) + x
        return x


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=[0.3, 0.2]):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.dropout = dropout
        self.dense1 = torch.nn.Linear(hidden_dim, 32)
        self.dense2 = torch.nn.Linear(32, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        x = x.unsqueeze(1)
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1], self.dropout[0])
        out = self.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        return out


class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """

    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.ReLU()
        # self.softmax=nn.Softmax(dim=-1)
        # if dropout_rate > 0.0:
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        # self.gru=nn.GRU(196,96,1,batch_first=True)
        # else:
        #     self.dropout1 = None
        #     self.dropout2 = None

    def forward(self, x):
        # out,h=self.gru(x,h)
        # out=F.dropout(out[:,-1],self.dropout1)
        out = self.fc1(x)
        out = self.act(out)
        # if self.dropout1:
        out = self.dropout1(out)
        out = self.fc2(out)
        # out = self.dropout2(out)
        # out=self.softmax(out)
        return out


class Accident(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim=384, depth=12, heads=6, pool='mean',
                 in_channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0., scale_dim=4, drop_path_rate=0, norm_layer=None):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        # patch_dim = in_channels * patch_size ** 2
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, dim),
        # )
        self.to_patch_embedding = PatchEmbed(img_size=image_size, patch_size=patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.space_transformer = Transformer(dim, 2, heads, dim_head, dim * scale_dim, dropout)
        # self.space_transformer = MHAttention(192,1)
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, 3, heads, dim_head, dim * scale_dim, dropout)
        # self.temporal_transformer = MHAttention(192,1)
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool
        self.h_dim = 96
        self.n_layers = 1
        mlp_ratio = 4.0,
        qkv_bias = True,
        qk_scale = None,
        drop_rate = 0.1,
        attn_drop_rate = 0.1,

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        ats_blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        num_tokens = [197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 197],
        drop_tokens = False,
        control_flags = [True for _ in range(depth)]
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens
        self.interval = 5
        self.blocks = []
        for i in range(depth):
            if i in self.ats_blocks[0]:
                self.blocks.append(
                    ATSBlock(
                        dim=dim,
                        num_heads=heads,
                        mlp_ratio=float(mlp_ratio[0]),
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=float(drop_rate[0]),
                        attn_drop=float(attn_drop_rate[0]),
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        insert_control_point=control_flags[i],
                        drop_tokens=drop_tokens,
                    )
                )
            else:
                self.blocks.append(
                    Block(
                        dim=dim,
                        num_heads=heads,
                        mlp_ratio=float(mlp_ratio[0]),
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=float(drop_rate[0]),
                        attn_drop=float(attn_drop_rate[0]),
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        insert_control_point=control_flags[i],
                    )
                )
        # self.blocks = nn.ModuleList(self.blocks)
        self.norm = norm_layer(dim)
        self.Maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp_head = MlpBlock(192, 64, 2)

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.triple_loss = TripletLoss(0.5)


    def get_feature(self, x):
        x = self.to_patch_embedding(x)
        # x=self.to_patch_embedding(x)
        b, t, n, _ = x.shape
        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        # x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # x = rearrange(x, 'b n d -> (b t) n d')
        # x = self.space_transformer(x)
        # idx
        init_n = x.shape[1]
        policy = torch.ones(b * t, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)
        # idx = 0
        for idx, blk in enumerate(self.blocks):
            blk.to(x.device)
            if idx in self.ats_blocks[0]:
                x, policy = blk(
                    x=x,
                    n_tokens=self.num_tokens[0][idx],
                    policy=policy,
                    sampler=sampler,
                    n_ref_tokens=init_n,
                )
                # idx += 1
                # policies.append(policy)
                # print(x.shape)
            else:
                x = blk(x=x, policy=policy, sampler=sampler)
                # print(x.shape)
                # policies.append(policy)
        # x=x.mean(dim=1)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)
        # x = rearrange(x, '(b t) ... -> b t ...', b=b)
        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = x.permute(0, 2, 1).contiguous()
        x = self.Maxpool(x).permute(0, 2, 1).contiguous()
        x = self.norm(x).squeeze(1)
        return x

    # def accident_loss(self, pred, target):
    #     target_cls = target[:, 1]
    #     target_cls = target_cls.to(torch.long)
    #     # penalty = -torch.max(torch.zeros_like(tai).to(pred.device),
    #     #                      (tai.to(pred.dtype) - time - 1) / fps)
    #     # pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
    #     neg_loss = self.ce_loss(pred, target_cls)
    #     neg_loss=torch.mean(neg_loss)
    #     # loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
    #     return  neg_loss
    #
    def pos_loss(self, pred, target, time, tai, fps=30.0):
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(tai).to(pred.device),
                             (tai.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls)).sum()
        # neg_loss = self.ce_loss(pred, target_cls)
        # loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        # pos_loss=torch.mean( pos_loss)
        return pos_loss

    def neg_loss(self, pred, target):
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        neg_loss = self.ce_loss(pred, target_cls).sum()
        return neg_loss

    def _exp_loss(self, pred, target, time, tai, fps=30.0):
        target_cls = target[:, 1]
        target_cls = target_cls.to(torch.long)
        penalty = -torch.max(torch.zeros_like(tai).to(pred.device),
                             (tai.to(pred.dtype) - time - 1) / fps)
        pos_loss = -torch.mul(torch.exp(penalty), -self.ce_loss(pred, target_cls))
        neg_loss = self.ce_loss(pred, target_cls)
        loss = torch.mean(torch.add(torch.mul(pos_loss, target[:, 1]), torch.mul(neg_loss, target[:, 0])))
        return loss

    def get_accident(self, x):
        x = self.mlp_head(x)
        # x = torch.unsqueeze(x, 1)
        # x, h = self.gru_net(x, h)
        return x

        # # computing losses
        # L1 = self._exp_loss(output, y, t, toa=toa, fps=10.0)
        # losses['total_loss'] += L1
        # all_output.append(output)  # TO-DO: all hidden

    def forward(self, triple_x, triple_tai, triple_label, start):
        ov = triple_x[0]
        nv = triple_x[1]
        av = triple_x[2]
        # ov_label = triple_label[0]
        nv_label = triple_label[1]
        av_label = triple_label[2]
        # ov_tai = triple_tai[0]
        nv_tai = triple_tai[1]
        av_tai = triple_tai[2]

        # all_label=torch.cat([nv_label, av_label],dim=0)
        # all_tai=torch.cat([ nv_tai,av_tai],dim=0)
        # start_ov = random.randint(0, 150 - 22)
        # start_ov=(start[0]//3)*3
        # start_nv = (start[0]//3)*3
        start_av = (start[1] // 3) * 3
        # start_ov=start[0]
        # start_nv = start[0]
        # start_av = start[1]
        # losses1=0
        # loss_pos=[]
        # loss_neg=[]
        loss_pos = 0
        loss_neg = 0
        # loss_t=[]
        triple_ov_f = torch.Tensor().to(ov.device)
        triple_nv_f = torch.Tensor().to(ov.device)
        triple_av_f = torch.Tensor().to(ov.device)
        # h = Variable(torch.zeros(self.n_layers, ov.size(0), self.h_dim)).to(ov.device)
        for t in range(0, ov.shape[1], 3):
            # print(t)
            ov_t = ov[:, t:t + 5, :, :, :]
            nv_t = nv[:, t:t + 5, :, :, :]
            av_t = av[:, t:t + 5, :, :, :]
            l = ov_t.shape[1]
            if l < 5:
                n = 5 - l
                repeat_frame0 = ov_t[:, -1, :, :, :].unsqueeze(1).repeat(1, n, 1, 1, 1)
                ov_t = torch.cat([ov_t, repeat_frame0], dim=1)
                repeat_frame1 = nv_t[:, -1, :, :, :].unsqueeze(1).repeat(1, n, 1, 1, 1)
                nv_t = torch.cat([nv_t, repeat_frame1], dim=1)
                repeat_frame2 = av_t[:, -1, :, :, :].unsqueeze(1).repeat(1, n, 1, 1, 1)
                av_t = torch.cat([av_t, repeat_frame2], dim=1)

            ov_f = self.get_feature(ov_t)
            nv_f = self.get_feature(nv_t)
            av_f = self.get_feature(av_t)

            # if t >= start_ov and t < start_ov + 22:
            #     triple_ov_f = torch.cat([triple_ov_f, ov_f], dim=0)
            # if t >= start_nv and t < start_nv + 22:
            #     triple_nv_f = torch.cat([triple_nv_f, nv_f], dim=0)
            if t >= start_av and t < start_av + 22:
                triple_ov_f = torch.cat([triple_ov_f, ov_f], dim=0)
                triple_nv_f = torch.cat([triple_nv_f, nv_f], dim=0)
                triple_av_f = torch.cat([triple_av_f, av_f], dim=0)
                # else:
                pass
            # out_ov = self.get_accident(ov_f)
            out_nv = self.get_accident(nv_f)
            # print("1",out_nv)
            out_av = self.get_accident(av_f)

            L_tp = self.pos_loss(out_av, av_label, t, av_tai, fps=30)
            L_tn = self.neg_loss(out_nv, nv_label)
            l= L_tp+0.7*L_tn
            loss_pos += l
        L1 = self.triple_loss(triple_ov_f, triple_nv_f, triple_av_f)
        loss1 =loss_pos+ 0.3 * L1

        return loss1
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
         super().__init__()
         #局部赋值使用
         self.dim = dim
         self.n_win = n_win
         self.num_heads =num_heads
         self.qk_dim = qk_dim or dim
         assert qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim维度必须能够整除注意力头数，否则无法形成完整多头q,v'
         self.scale =qk_scale or self.qk_dim ** -0.5#就是注意力计算公式中的缩放因子可以自己设计也可以直接等于维度的1/2次方的倒数

         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)#通过深度可分离卷积，对输出维度和输入维度相同的数据进行卷积，判定为否，赋值x元素全零
         #全局统一赋值使用
         self.topk = topk
         self.param_routing = param_routing
         self.diff_routing = diff_routing
         self.soft_routing = soft_routing
         assert not (self.param_routing and not self.diff_routing)#确保两个参数使用相同数值

         # 返回两个一个是Top-k索引，一个是Tok-k的权重注意得分
         self.router = TopkRouting(qk_dim=self.qk_dim,
                                   qk_scale=self.scale,
                                   topk=self.topk,
                                   diff_routing=self.diff_routing,
                                   param_routing=self.param_routing)

         if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
         elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
         else:  # hard non-differentiable routing
            mul_weight = 'none'
         self.kv_gather = KVGather(mul_weight=mul_weight)#从kv张量中选择topk

         # qkv 映射（全局路由和局部注意力共享）
         self.param_attention = param_attention
         if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
         elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
         else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

         self.kv_down = nn.Identity()

         self.attn_act = nn.Softmax(dim=-1)
         self.auto_pad=auto_pad
    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        if self.auto_pad:#是否自动补全，可以适应多尺度图像进行训练
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        #路由检索，找到按照窗口的QK前topK值及其索引
        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors
        #通过索引找到kv里面对应的
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe#连接
        # output linear
        out = self.wo(out)#nn.Identity
        #最后去除刚才自动补全的像素
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out





class TopkRouting(nn.Module):
    def __init__(self, qk_dim, qk_scale = None, topk = 4, diff_routing = False, param_routing = False):
         super().__init__()
         self.topk = topk
         self.qk_dim = qk_dim
         self.scale = qk_scale or qk_dim ** -0.5
         self.diff_routing = diff_routing
         self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()#对
         self.routing_act = nn.Softmax(dim=-1)#路由激活函数，以最后一个维度

    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:#返回两个合并成元组类型张量
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()#返回一个相同query.key参数,原始的query，key将不再会改变
        query_hat, key_hat = self.emb(query), self.emb(key)  # 全连接 per-window pooling -> (n, p^2, c) 其实是nn.Identity
        attn_logit = ( query_hat * self.scale) @ key_hat.transpose(-2,-1) #定义原始# (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)#最后一个维度按照行进行 索引
        #￥￥￥￥￥￥￥￥￥核心代码
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        return r_weight, topk_index# (n, p^2, k), (n, p^2, k)

#从kv张量中选择topk
class KVGather(nn.Module):
    def __init__(self,mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight


    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv
        # q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        # return q, k, v


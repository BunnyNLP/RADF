import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, query_input_dim, key_input_dim, value_input_dim, query_hidden, key_hidden, value_hidden, output_dim):
        super().__init__()

        assert query_input_dim == key_input_dim
        self.n_head = n_head
        self.d_q = query_hidden
        self.d_k = key_hidden
        self.d_v = value_hidden

        self.fc_q = nn.Linear(query_input_dim, n_head * query_hidden)
        self.fc_k = nn.Linear(key_input_dim, n_head * key_hidden)
        self.fc_v = nn.Linear(value_input_dim, n_head * value_hidden)

        self.attention = ScaledDotProductAttention(scale=np.power(key_hidden, 0.5))

        self.fc_o = nn.Linear(n_head * value_hidden, output_dim)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_q, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        assert n_k == n_v

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
          mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output

class SelfAttention(nn.Module):
    """ Self-Attention """
    
    def __init__(self, n_head, d_k, d_v, d_x, d_o):
        super().__init__()
        self.wq = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wk = nn.Parameter(torch.Tensor(d_x, d_k))
        self.wv = nn.Parameter(torch.Tensor(d_x, d_v))

        self.mha = MultiHeadAttention(n_head=n_head, d_k_=d_k, d_v_=d_v, d_k=d_k, d_v=d_v, d_o=d_o)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / np.power(param.size(-1), 0.5)
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask=None):
        q = torch.matmul(x, self.wq)   
        k = torch.matmul(x, self.wk)
        v = torch.matmul(x, self.wv)

        attn, output = self.mha(q, k, v, mask=mask)


        return attn, output


if __name__ =='__main__':
    # self-attention
    batch = 16
    n_x = 256
    d_x = 1024
        
    x = torch.randn(batch, n_x, d_x)
    mask = torch.zeros(batch, n_x, n_x).bool()

    selfattn = SelfAttention(n_head=8, d_k=1024, d_v=1024, d_x=d_x, d_o=d_x)
    attn, output = selfattn(x, mask=mask)
    print(mask.shape)#torch.Size([16, 256, 256])
    print(x.shape)#torch.Size([16, 256, 1024])
    print(attn.size())#torch.Size([128, 256, 256])
    print(output.size())#torch.Size([16, 256, 1024])
    


    mha = MultiHeadAttention(n_head=2, d_k_=1024, d_v_=1024, d_k=1024, d_v=1024, d_o=d_x)
    q = torch.rand(batch,41,d_x)
    k = x
    v = x
    attn ,output = mha(q,k,v)
    print(mask.shape)#torch.Size([16, 256, 256])
    print(x.shape)#torch.Size([16, 256, 1024])
    print(attn.size())#torch.Size([32, 41, 256])
    print(output.size())#torch.Size([16, 41, 1024])



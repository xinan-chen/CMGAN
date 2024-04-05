import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# source: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


# attention, feedforward, and conv module


class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, max_pos_emb=512):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, context_mask=None):
        n, device, h, max_pos_emb, has_context = (
            x.shape[-2],
            x.device,
            self.heads,
            self.max_pos_emb,
            exists(context),
        )
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device=device)
        dist = rearrange(seq, "i -> i ()") - rearrange(seq, "j -> () j")
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum("b h n d, n r d -> b h n r", q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device=device))
            context_mask = (
                default(context_mask, mask)
                if not has_context
                else default(
                    context_mask, lambda: torch.ones(*context.shape[:2], device=device)
                )
            )
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, "b i -> b () i ()") * rearrange(
                context_mask, "b j -> b () () j"
            )
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return self.dropout(out)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout), # 相当于没有dropout
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(
        self, dim, causal=False, expansion_factor=2, kernel_size=31, dropout=0.0
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange("b n c -> b c n"),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(
                inner_dim, inner_dim, kernel_size=kernel_size, padding=padding
            ),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange("b c n -> b n c"),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# Conformer Block


class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0.0,
        ff_dropout=0.0,
        conv_dropout=0.0
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(
            dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout
        )
        self.conv = ConformerConvModule(
            dim=dim,
            causal=False,
            expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.ff1(x) + x
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
    

# RNN


class DPRNN(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN, self).__init__(**kwargs)
        self.numUnits = numUnits
        
        self.intra_rnn = nn.GRU(input_size = self.numUnits, hidden_size = self.numUnits//2, batch_first = True, bidirectional = True)
    
        self.intra_fc = nn.Linear(self.numUnits, self.numUnits)

        self.intra_ln = nn.InstanceNorm2d(width,eps=1e-8)

        # self.inter_rnn = nn.GRU(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        
        # self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        # self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

        self.width = width
        self.channel = channel
    
    def forward(self,x, bs=1):

        if not x.is_contiguous(): # 检查是否连续
            x = x.contiguous()   
        ## Intra RNN    
        # inp = x.permute(0,2,3,1).contiguous() 
        # intra_LSTM_input = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2], inp.shape[3]) #(Bs*T, F, C)
        intra_LSTM_out = self.intra_rnn(x)[0] #(Bs*T, F, C) hidden_size*num_directions == C
        intra_dense_out = self.intra_fc(intra_LSTM_out)
        intra_ln_input = intra_dense_out.view(bs, -1, self.width, self.channel) #(Bs, T, F, C) ?????
        intra_ln_input = intra_ln_input.permute(0,2,1,3) #(Bs, F, T, C)
        intra_out = self.intra_ln(intra_ln_input) 
        intra_out = intra_out.permute(0,2,1,3) #(Bs, T, F, C)
        intra_out = intra_out.view(-1, intra_out.shape[2],intra_out.shape[3])#(Bs*T, F, C)
        intra_out = torch.add(x, intra_out) 
        ## Inter RNN
        # inter_LSTM_input = x.permute(0,3,2,1) #(Bs, F, T, C)
        # inter_LSTM_input = inter_LSTM_input.contiguous()
        # inter_LSTM_input = inter_LSTM_input.view(inter_LSTM_input.shape[0] * inter_LSTM_input.shape[1], inter_LSTM_input.shape[2], inter_LSTM_input.shape[3]) #(Bs * F, T, C)
        # inter_LSTM_out = self.inter_rnn(inter_LSTM_input)[0]
        # inter_dense_out = self.inter_fc(inter_LSTM_out)
        # inter_dense_out = inter_dense_out.view(x.shape[0], self.width, -1, self.channel) #(Bs, F, T, C)
        # inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        # inter_out = self.inter_ln(inter_ln_input)
        # # inter_out = inter_out.permute(0,2,3,1) #(Bs, T, F, C)
        # inter_out = torch.add(x, inter_out)
        # # inter_out = inter_out.permute(0,3,1,2)
        # inter_out = inter_out.contiguous()
        
        return intra_out

class DPRNN_t(nn.Module):
    def __init__(self, numUnits, width, channel, **kwargs):
        super(DPRNN_t, self).__init__(**kwargs)
        self.numUnits = numUnits

        self.inter_rnn = nn.GRU(input_size = self.numUnits, hidden_size = self.numUnits, batch_first = True, bidirectional = False)
        
        self.inter_fc = nn.Linear(self.numUnits, self.numUnits)
        
        self.inter_ln = nn.InstanceNorm2d(channel, eps=1e-8)

        self.width = width
        self.channel = channel
    
    def forward(self, x):

        if not x.is_contiguous():
            x = x.contiguous()    # (Bs*F, T, C)
        bs_F, T, C = x.size()
        bs = bs_F // self.width
        ## Inter RNN
        inter_LSTM_out = self.inter_rnn(x)[0]
        inter_dense_out = self.inter_fc(inter_LSTM_out)
        inter_dense_out = inter_dense_out.view(bs, self.width, -1, self.channel) #(Bs, F, T, C)
        inter_ln_input = inter_dense_out.permute(0,3,2,1) #(Bs, C, T, F)
        inter_out = self.inter_ln(inter_ln_input)
        inter_out = inter_out.permute(0, 3, 2, 1).contiguous().view(bs_F, T, C)# (Bs*F, T, C)
        inter_out = torch.add(x, inter_out)
        inter_out = inter_out.contiguous()
        
        return inter_out
    

class CRNNBlock(nn.Module):
    def __init__(
        self,
        *, # 强制后面的参数必须使用关键字传递
        dim,
        width,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.intra_rnn = DPRNN(dim,width,dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # self.attn = PreNorm(dim, self.attn)
        self.intra_rnn = PreNorm(dim, self.intra_rnn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        # x = self.attn(x, mask = mask) + x
        x = self.intra_rnn(x)
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
    
class CRNNBlock_t(nn.Module):
    def __init__(
        self,
        *,
        dim,
        width,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.intra_rnn = DPRNN_t(dim, width, dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        # self.attn = PreNorm(dim, self.attn)
        self.intra_rnn = PreNorm(dim, self.intra_rnn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        # x = self.attn(x, mask = mask) + x
        x = self.intra_rnn(x)
        x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
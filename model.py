import dataclasses
import functools
import math
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

# SHAPE[0] IS DIM ATT
def time_first_init(shape, dtype):
    """Initialises time_first parameter of the TimeMix module. Shape is assumed to be 1d and equal to att_dim."""
    return jnp.ones(shape, dtype=dtype) * math.log(0.3) + jnp.array([(i + 1) % 3 - 1 for i in range(shape[0])], dtype=dtype) * 0.5

# I could use classes with __call__ method but meh?
def time_decay_init(layer_id, num_layers):
    """Initialises time_decay paramater of the TimeMix module. Shape is assumed to be 1d and equal to att_dim."""
    ratio_0_to_1 = layer_id / (num_layers - 1)
    def init(shape, dtype):
        # decay_speed = jnp.ones(shape, dtype=dtype)
        # The only variable is h which is 0..n... see comment at the bottom, I wrote it first.
        # for h in range(shape[0]):
        #     decay_speed = decay_speed.at[h].set(-5 + 8 * (h / (shape[0] - 1)) ** (0.7 + 1.3 * ratio_0_to_1))
        # return decay_speed
        return (-5 + 8 * ( jnp.arange(shape[0], dtype=dtype) / (shape[0] - 1) ) ** (0.7 + 1.3 * ratio_0_to_1))
    return init

def get_ddd(shape, dtype):
    """Get's the ddd const required for intialising time_mix_k/v/r."""
    # I think there was a function that did 0..n?
    # ddd = jnp.ones(shape, dtype)
    # for i in range(shape[-1]):
    #     ddd = ddd.at[0, i].set(i / shape[-1])
    # return ddd
    return jnp.arange(shape[-1], dtype=dtype)[jnp.newaxis,:] / shape[-1]

def time_mix_k_init(layer_id, num_layers):
    """Initialises time_mix_k paramter of the TimeMix module. Also used for time_mix_v."""
    ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
    def init(shape, dtype):
        return jnp.power(get_ddd(shape, dtype), ratio_1_to_almost0)
    return init

def time_mix_v_init(layer_id, num_layers):
    """Initialises time_mix_v parameter of the TimeMix module."""
    ratio_0_to_1 = layer_id / (num_layers - 1)
    tmkf = time_mix_k_init(layer_id, num_layers)
    def init(shape, dtype):
        return tmkf(shape, dtype) + 0.3 * ratio_0_to_1
    return init

def time_mix_r_init(layer_id, num_layers):
    """Initialises time_mix_r parameter of the TimeMix module."""
    ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
    def init(shape, dtype):
        return jnp.power(get_ddd(shape, dtype), ratio_1_to_almost0 * 0.5)
    return init

def time_shift_pad(x):
    # tfw no... I mean ZeroPad2d
    # format is (top, bottom) * dims?
    # Shape should be T, C? (jax.vmap already handles the funny B dim)
    return jnp.pad(x, [(1, 0), (0, 0)])[:-1, :]

# This is my nightmare, it is also numerically unstable, I just wasn't able to reproduce the results after a while...
# decay, first, k, v
@jax.jit
def WKV_(w: jnp.ndarray, u: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray):
    # Ok, so knowing a bit how RNN works...
    # I SPECIFICALLY SAID IN THE FUCKING POLICY TO COMPUTE IN f32
    dtype = k.dtype
    w, u, k, v = w.astype(jnp.float32), u.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)

    # if '.time_decay' in x:
    #     w[x] = w[x].float()
    #     w[x] = -torch.exp(w[x])
    w = -jnp.exp(w)

    # if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
    #     w[x] = w[x].t()
    k = k.T
    v = v.T

    # ww = time_first + k
    # p = torch.maximum(pp, ww)
    # e1 = torch.exp(pp - p)
    # e2 = torch.exp(ww - p)
    # b = e1 * bb + (e2)
    k = jnp.exp(k)
    # a = e1 * aa + (e2 * v)
    a = k * v

    # e1 * aa/bb is more complicated because it relies on rolling state?
    
    # Thanks to https://github.com/tensorpro/jax-rwkv/blob/2e357089aa1e46e32cbb333575580eae3adbd525/jax-rwkv/model.py#L32 for helping me understand it a bit better

    # ok so, time_decay to k will happen for every token at the end, so we have 0..T-1 times the time_decay for every token in reverse
    # reversing 0..T-1 will give us T-2..=0. and we also do time_first
    T = k.shape[1]
    # exp of w over time
    time_decay_k = jnp.arange(-(T-2), 1) * jnp.exp(w)[:, jnp.newaxis]
    # print(time_decay_k.shape)
    # Complete time
    time_decay_k = jnp.concatenate([time_decay_k, u[:, jnp.newaxis]], axis=1)
    # print(time_decay_k.shape)
    # Exp of complete time, so we get the full decay for the convolution
    time_decay_k = jnp.exp(time_decay_k)[:, jnp.newaxis]
    # print(time_decay_k.shape)

    # now my brain doesn't work and I forgot the line of thought I had...

    # I think convolution is what I want? It's that fancy fucking thing that computes how shape changes over time
    # feature_group_count 	int64 	the number of feature groups
    # We have a total of C aka numvbr of tokens features
    C = k.shape[0]
    # https://github.com/deepmind/dm-haiku/blob/a65f7db911d6c1821cb6befb28aaa519714b62e5/haiku/_src/conv.py#L205
    # This is required because convolution requires n+2 rank, and so far we only did n+1?
    a = a[jnp.newaxis, :]
    k = k[jnp.newaxis, :]
    # print(a.shape, k.shape)
    # TODO: check if precision really matters...
    a = jax.lax.conv_general_dilated(jnp.pad(a, [(0, 0), (0, 0), (T-1, 0)]), time_decay_k, (1,), [(0, 0)], feature_group_count=C, precision=jax.lax.Precision.HIGHEST)
    b = jax.lax.conv_general_dilated(jnp.pad(k, [(0, 0), (0, 0), (T-1, 0)]), time_decay_k, (1,), [(0, 0)], feature_group_count=C, precision=jax.lax.Precision.HIGHEST)

    # now you can clown upon me
    # a[a==0] = jnp.finfo(jnp.float32).eps
    # b = b.at[b==0].set(jnp.finfo(jnp.float32).max)
    # b = b.at[b==0].set(1)
    # print(a, b, jnp.any(a == 0), jnp.any(b == 0))

    # I am not going to double transpose the matrix, you won't make me
    return (a/b)[0].astype(dtype)

# Foil Inc.
# Now defunct
@jax.jit
def WKV(w: jnp.ndarray, u: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray):
    T, C = k.shape
    # print(k.shape)

    dtype = k.dtype
    w, u, k, v = w.astype(jnp.float32), u.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)

    w = -jnp.exp(w)
    # k = k.T
    # v = v.T
    # k = k[:, jnp.newaxis]
    # v = v[:, jnp.newaxis]
    # print(k.shape, v.shape)
    sl = []
    s = 2
    while s <= T:
        sl += [(s, (s >> 1) - 1, s - 1, T - T % s)]
        s = s << 1
    s = s >> 1
    while s >= 2:
        sl += [(s, s - 1, (s >> 1) * 3 - 1, T - (T % s < (s >> 1)) * (s >> 1))]
        s = s >> 1

    oo = k.copy()
    pp = v.copy()
    qq = jnp.ones((T, C), dtype=w.dtype)
    dd = jnp.ones((T, 1), dtype=w.dtype)
    for ss, sa, sb, sz in sl:
        p = pp[sb:sz:ss]
        if p.shape[0] == 0:
            continue
        q = qq[sb:sz:ss]
        d = dd[sb:sz:ss]
        o = oo[sb:sz:ss]
        e = oo[sa:sz:ss] + d * w
        x = jnp.maximum(e, o)
        a = jnp.exp(e - x)
        b = jnp.exp(o - x)
        # print('inner', p.shape, pp.shape)
        pp = pp.at[sb:sz:ss].set(a * pp[sa:sz:ss] + b * p)
        qq = qq.at[sb:sz:ss].set(a * qq[sa:sz:ss] + b * q)
        dd = dd.at[sb:sz:ss].set(dd[sa:sz:ss] + d)
        oo = oo.at[sb:sz:ss].set(x)

    p = jnp.roll(pp, 1, axis=0)
    q = jnp.roll(qq, 1, axis=0)
    o = jnp.roll(oo, 1, axis=0)
    # print(p, q, o, pp.shape, p.shape, qq.shape, q.shape, oo.shape, o.shape, k.shape, v.shape)
    # (1024, 2) (1024, 2)
    # (2, 1, 1024) (2, 1, 1024)
    # (1024, 2) (1024, 2)

    # print(o.shape, k.shape, v.shape, u.shape)
    # x = jnp.maximum(o, k + u[jnp.newaxis, jnp.newaxis, :])
    x = jnp.maximum(o, k + u[jnp.newaxis, :])
    a = jnp.exp(o - x)
    b = jnp.exp(k + u - x)
    y = (a * p + b * v) / (a * q + b)
    # print(v.shape, y.shape, v, y, sl)
    # print(v.shape, y.shape)
    # y = jnp.concatenate([v[:1, :, :], y[1:, :, :]]) # shapes=[(1024, 2), (1024,)]
    y = jnp.concatenate([v[:1, :], y[1:, :]]) # shapes=[(1024, 2), (1024,)]
    # y = y.T
    # y = y.swapaxes(1, 0)
    # return y[0].T.astype(dtype)
    return y.T.astype(dtype)

# Perhaps my original vmap idea was correct? because we can only process one C at a time, I just got the axi incorrect (it was 0, not 1)
# But then again how the hell do I properly pass it inside the scan'd func?
# ergh... we also have k&v of shapes (T, C)
# if we apply the C vmap then we will get shapes of:
#  * w and u being scalar
#  * k and v being T 1d vectors
# WORKS BUT DOES NOT MATCH!!!
@functools.partial(jax.vmap, in_axes=(0, 0, 1, 1), out_axes=1)
# @jax.jit
def WKV_n(w, u, k, v):
    dtype = k.dtype
    w, u, k, v = w.astype(jnp.float32), u.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    w = -jnp.exp(w)
    # print(w, u, '---', w.shape, u.shape, k.shape, v.shape)
    # print(w.shape, u.shape, k.shape, v.shape)
    # This should operate over T, idk how to pass `w, u`
    def body(carry, elems):
        # print('c', carry, 'e', elems)
        p, q, o = carry
        k, v = elems
        # print(k, v, k.shape, v.shape, u.shape, w.shape)
        
        no = jnp.maximum(o, u + k)
        A = jnp.exp(o - no)
        B = jnp.exp(u + k - no)
        y = (A * p + B * v) / (A * q + B)

        no = jnp.maximum(w + o, k)
        A = jnp.exp(w + o - no)
        B = jnp.exp(k - no)
        p = A * p + B * v
        q = A * q + B
        o = no
        return ((p, q, o), y)
    # But at what fucking point do I apply vmap?
    # CUDA core iterates over T, one C at a time. y produced is valid for only one C and spans over T
    # concat makes so we get pair of (k, v)
    # I don't take [-1] here so it can be reused for getting the hidden state...
    return jax.lax.scan(body, (0.0, 0.0, -1e38), jnp.concatenate([k[:, jnp.newaxis], v[:, jnp.newaxis]], axis=-1)).astype(dtype)


@dataclasses.dataclass
class Attention(hk.Module):
    """RWKV Attention block aka TimeMix."""

    layer_id: int
    num_layers: int
    n_embd: int
    dim_att: int
    name: Optional[str] = None

    # done through a policy
    # calc_dtype: np.dtype = jnp.float32

    # def __post_init__(self):
    #     ratio_0_to_1 = layer_id / (num_layers - 1)
    #     ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)
    
    def __call__(self, x):
        time_first = hk.get_parameter("time_first", [self.dim_att], init=time_first_init)
        time_decay = hk.get_parameter("time_decay", [self.dim_att], init=time_decay_init(self.layer_id, self.num_layers))

        time_mix_k = hk.get_parameter("time_mix_k", [1, self.dim_att], init=time_mix_k_init(self.layer_id, self.num_layers))
        time_mix_v = hk.get_parameter("time_mix_v", [1, self.dim_att], init=time_mix_v_init(self.layer_id, self.num_layers))
        time_mix_r = hk.get_parameter("time_mix_r", [1, self.dim_att], init=time_mix_r_init(self.layer_id, self.num_layers))

        # Input size in Haiku is infered at first pass time
        key = hk.Linear(self.dim_att, with_bias=False, name="key")
        value = hk.Linear(self.dim_att, with_bias=False, name="value")
        receptance = hk.Linear(self.dim_att, with_bias=False, name="receptance")
        output = hk.Linear(self.n_embd, with_bias=False, name="output")

        # --- Calc ---

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = time_shift_pad(x)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xv = x * time_mix_v + xx * (1 - time_mix_v)
        xr = x * time_mix_r + xx * (1 - time_mix_r)
        # Can I just not do the 'a' aka b models...
        # TODO(mrsteyk): QKV timemix

        k = key(xk)
        v = value(xv)
        r = receptance(xr)
        sr = jax.nn.sigmoid(r)

        # print(np.any(np.isinf(xx)), np.any(np.isinf(xk)), np.any(np.isinf(k)))

        # TODOne(mrsteyk): CUDA core reimplement... why...
        # I did it in the end...
        # print(k.shape, v.shape)
        # wkv = WKV_n(time_decay[:, jnp.newaxis], time_first[:, jnp.newaxis], k, v).T
        wkv = WKV_n(time_decay, time_first, k, v)[-1]
        # print(wkv)
        # wkv = WKV(time_decay, time_first, k, v).T
        rwkv = sr * wkv
        rwkv = output(rwkv)
        # print(f"att{self.layer_id}", rwkv)
        return rwkv

@dataclasses.dataclass
class FFN(hk.Module):
    """RWKV FFN block, also known as ChannelMix."""

    layer_id: int
    num_layers: int
    n_embd: int
    dim_ffn: int
    name: Optional[str] = None

    def __call__(self, x):
        time_mix_k = hk.get_parameter("time_mix_k", [1, self.n_embd], init=time_mix_k_init(self.layer_id, self.num_layers))
        time_mix_r = hk.get_parameter("time_mix_r", [1, self.n_embd], init=time_mix_k_init(self.layer_id, self.num_layers))

        key = hk.Linear(self.dim_ffn, with_bias=False, name="key")
        receptance = hk.Linear(self.n_embd, with_bias=False, name="receptance")
        value = hk.Linear(self.n_embd, with_bias=False, name="value")

        # --- Calc ---
        xx = time_shift_pad(x)
        xk = x * time_mix_k + xx * (1 - time_mix_k)
        xr = x * time_mix_r + xx * (1 - time_mix_r)

        k = key(xk)
        k = jnp.square(jax.nn.relu(k))
        kv = value(k)
        rkv = jax.nn.sigmoid(receptance(xr)) * kv
        # print(f"ffn{self.layer_id}", x.shape, x, rkv)
        return rkv

@dataclasses.dataclass
class Block(hk.Module):
    """RWKV block."""

    layer_id: int
    num_layers: int
    n_embd: int
    dim_att: int
    dim_ffn: int
    name: Optional[str] = None

    def __call__(self, x):
        ln1 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="ln1")
        ln2 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="ln2")
        att = Attention(layer_id=self.layer_id, num_layers=self.num_layers, n_embd=self.n_embd, dim_att=self.dim_att, name="att")
        ffn = FFN(layer_id=self.layer_id, num_layers=self.num_layers, n_embd=self.n_embd, dim_ffn=self.dim_ffn, name="ffn")

        # --- Calc ---

        # You are smart (C) CrossProduct
        xx = ln1(x)
        # print(x) # matches pure jax
        x = x + att(xx)
        x = x + ffn(ln2(x))

        # TODO(mrsteyk): tiny QKV ATT

        return x

@dataclasses.dataclass
class RWKV(hk.Module):
    """RWKV module."""

    num_layers: int
    vocab_size: int
    n_embd: int
    dim_att: int
    dim_ffn: int
    name: Optional[str] = None

    def __call__(self, x):
        emb = hk.Embed(vocab_size=self.vocab_size, embed_dim=self.n_embd, name="emb")
        ln0 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="ln0")
        blocks = [Block(layer_id=i, num_layers=self.num_layers, n_embd=self.n_embd, dim_att=self.dim_att, dim_ffn=self.dim_ffn, name=f"block_{i}") for i in range(self.num_layers)]
        ln_out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="ln_out")
        head = hk.Linear(self.vocab_size, with_bias=False, name="head")

        x = ln0(emb(x))
        for block in blocks:
            x = block(x)
        x = ln_out(x)
        return head(x)
import dataclasses
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

# This is my nightmare
# decay, first, k, v
@jax.jit
def WKV(w, u, k, v):
    # Ok, so knowing a bit how RNN works...

    # if '.time_decay' in x:
    #     w[x] = w[x].float()
    #     w[x] = -torch.exp(w[x])
    w = -jnp.exp(w)

    # if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'output.weight' in x:
    #     w[x] = w[x].t()
    k, v = k.T, v.T

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
    time_decay_k = jnp.arange(-(T-2), 1)[jnp.newaxis, :] * jnp.exp(w)[:, jnp.newaxis]
    # print(time_decay_k.shape)
    time_decay_k = jnp.concatenate([time_decay_k, u[:,jnp.newaxis]], axis=1)
    # print(time_decay_k.shape)
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
    a = jax.lax.conv_general_dilated(jnp.pad(a, [(0,0)] * 2 + [(T-1, 0)]), time_decay_k, (1,), [(0, 0)], feature_group_count=C)
    b = jax.lax.conv_general_dilated(jnp.pad(k, [(0,0)] * 2 + [(T-1, 0)]), time_decay_k, (1,), [(0, 0)], feature_group_count=C)

    # I am not going to double transpose the matrix, you won't make me
    return (a/b)[0]

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

        # TODOne(mrsteyk): CUDA core reimplement... why...
        # I did it in the end...
        wkv = WKV(time_decay, time_first, k, v).T
        rwkv = sr * wkv
        return output(rwkv)

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
        return jax.nn.sigmoid(receptance(xr)) * kv

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

        x = x + att(ln1(x))
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
        ln0 = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        blocks = [Block(layer_id=i, num_layers=self.num_layers, n_embd=self.n_embd, dim_att=self.dim_att, dim_ffn=self.dim_ffn, name=f"block_{i}") for i in range(self.num_layers)]
        ln_out = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True, name="ln_out")
        head = hk.Linear(self.vocab_size, with_bias=False, name="head")

        x = ln0(emb(x))
        for block in blocks:
            x = block(x)
        x = ln_out(x)
        return head(x)
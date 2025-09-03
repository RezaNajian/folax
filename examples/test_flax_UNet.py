import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import jax
import optax
import math
import random as r
import numpy as np
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from flax_nnx_modules import OpenAI_UNetModel
import timeit
import statistics

rand_key = jax.random.PRNGKey(0)  

flax_nnx_UNetModel = OpenAI_UNetModel(image_size=384,
                                      in_channels=1,
                                      model_channels=128,
                                      out_channels=1, 
                                      num_res_blocks=2,
                                      attention_resolutions=tuple("32,16,8"),
                                      dropout=0,
                                      channel_mult=(1, 1, 2, 3, 4),
                                      num_heads=4,
                                      num_head_channels=64,
                                      num_heads_upsample=-1,
                                      use_scale_shift_norm=False,
                                      resblock_updown=False,
                                      use_new_attention_order=False,
                                      rngs=nnx.Rngs(default=rand_key))

# evaluate total number of trainable params
params = nnx.state(flax_nnx_UNetModel, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"Total number of trainable parameters:{total_params}")

load_param_from_pytorch = True
if load_param_from_pytorch:
  # read pytorch params 
  import pickle
  with open("/home/reza/jax-ai-stack/CoNFiLD/inference_phy_random_sensor/pytorch_params.pkl", "rb") as f:
    pytorch_params_dict = pickle.load(f) 

  graphdef, params = nnx.split(flax_nnx_UNetModel, nnx.Param)
  flat_nnx_params = dict(nnx.to_flat_state(params))

  for key,nnx_var in flat_nnx_params.items():

    param_name = str(key[0])
    for key_seg in key[1:]:
      if key_seg != 'layers':
        if key_seg in ["kernel","scale"]:
          param_name += str("."+str("weight"))
        else:
          param_name += str("."+str(key_seg))

    if param_name not in pytorch_params_dict.keys():
      print(f"could not find param with name {param_name} in pytorch_params_dict, EXIT !")
      exit()
    py_torch_param_shape = pytorch_params_dict[param_name].shape
    flax_param_shape = nnx_var.value.shape
    if py_torch_param_shape==flax_param_shape:
      nnx_var.value = pytorch_params_dict[param_name]
    elif pytorch_params_dict[param_name].T.shape == flax_param_shape:
      nnx_var.value = pytorch_params_dict[param_name].T
    else:
      print(f"param:{param_name}, has pytorch shape {py_torch_param_shape} and jax shape {flax_param_shape}, and are npt compatibel ! EXIT")
      exit()
  # now update
  nnx.update(flax_nnx_UNetModel, params) 


set_all_params_constant = False
if set_all_params_constant:
  graphdef, params = nnx.split(flax_nnx_UNetModel, nnx.Param)
  flat_nnx_params = dict(nnx.to_flat_state(params))    
  total_num_param_to_be_set = 0
  for key,nnx_var in flat_nnx_params.items():
    nnx_var.value = 0.01 * jnp.ones_like(nnx_var.value)
    total_num_param_to_be_set += 1    
  print(f"total_num_param_to_be_set:{total_num_param_to_be_set}")
  nnx.update(flax_nnx_UNetModel, params) 

CH = 1
BS = 10
H = 32
W = 32

np.random.seed(42)
batch_x = jnp.array(np.random.rand(BS,H,W,CH))
batch_t = jnp.array(np.random.rand(BS))  

print(f"jnp.sum(batch_x):{jnp.sum(batch_x)},jnp.sum(batch_t):{jnp.sum(batch_t)}")

output = flax_nnx_UNetModel(batch_x.reshape(BS,H,W,CH),batch_t.reshape(BS))
print(output.shape)
print(jnp.sum(output))

@jax.jit
def infer(batch_x,batch_t):
  return flax_nnx_UNetModel(batch_x,batch_t)

#warm up
output = infer(batch_x,batch_t).block_until_ready()
print(output.shape)
print(jnp.sum(output))

# Benchmark Inference
n_repeat = 20   # How many times to repeat the timing
n_number = 20    # How many times to run the function in each repeat

times = timeit.repeat(lambda: infer(batch_x,batch_t).block_until_ready(), 
                      repeat=n_repeat, number=n_number)
normalized_times = np.array(times) / n_number

print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")

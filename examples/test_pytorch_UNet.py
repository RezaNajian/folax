import torch
import numpy as np
import sys
from pytorch_modules.unet import UNetModel
import random
import pickle
import timeit
import statistics

# Fix seeds for reproducibility
torch.manual_seed(42)       # For CPU
torch.cuda.manual_seed(42)  # For current GPU
torch.cuda.manual_seed_all(42)  # For all GPUs
np.random.seed(42)
random.seed(42)

# Optional: make CuDNN deterministic 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def set_model_params_random(module):
    total_num_param_to_be_set = 0
    for name, param in module.named_parameters():
        if param.requires_grad:       # only modify trainable parameters
            param.data.fill_(0.03*np.random.rand()) 
            total_num_param_to_be_set += 1
    return module

def export_model_params(module):
    param_dict = {}
    for name, param in module.named_parameters():
        if param.requires_grad:       # only modify trainable parameters
            param_dict[name] = param.data.numpy()

    with open("pytorch_params.pkl", "wb") as f:
        pickle.dump(param_dict, f)                

if torch.cuda.is_available():
  dev = "cuda"
else:
  dev = "cpu"

device = torch.device(dev)

pytorch_UNetModel= UNetModel(image_size=384,
                            in_channels=1,
                            model_channels=128,
                            out_channels=1, 
                            num_res_blocks=2,
                            attention_resolutions=tuple("32,16,8"),
                            dropout=0,
                            channel_mult=(1, 1, 2, 3, 4),
                            num_classes=None,
                            use_checkpoint=False,
                            use_fp16=False,
                            num_heads=4,
                            num_head_channels=64,
                            num_heads_upsample=-1,
                            use_scale_shift_norm=False,
                            resblock_updown=False,
                            use_new_attention_order=False)

pytorch_UNetModel = set_model_params_random(pytorch_UNetModel) 
export_model_params(pytorch_UNetModel)
pytorch_UNetModel.to(device)
pytorch_UNetModel.eval()

# Total number of trainable parameters
total_params = sum(p.numel() for p in pytorch_UNetModel.parameters() if p.requires_grad)
print("Total number of trainable parameters:", total_params)

CH = 1
BS = 10
H = 32
W = 32

batch_x = np.random.rand(BS,CH,H,W)
batch_t = np.random.rand(BS)  
batch_x = torch.from_numpy(batch_x).float()
batch_t = torch.from_numpy(batch_t).float()
batch_x = batch_x.to(device)
batch_t = batch_t.to(device)

print(f"torch.sum(batch_x):{torch.sum(batch_x)},torch.sum(batch_t):{torch.sum(batch_t)}")

output = pytorch_UNetModel(batch_x,batch_t)
print(output.shape)
print(torch.sum(output))

def infer(batch_x,batch_t):
    output = pytorch_UNetModel(batch_x,batch_t)
    return torch.sum(output)

# Benchmark Inference
n_repeat = 20   # How many times to repeat the timing
n_number = 20    # How many times to run the function in each repeat

times = timeit.repeat(lambda: infer(batch_x,batch_t), 
                      repeat=n_repeat, number=n_number)
normalized_times = np.array(times) / n_number

print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")


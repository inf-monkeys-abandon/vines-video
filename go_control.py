import torch
from model import Model
from utils import USE_OUR_METHOD_FLAG
import os
from time import time

USE_OUR_METHOD_FLAG.set(False)

in_prompt = 'an astronaut is dancing'
out_name = in_prompt.replace(' ', '_').replace('.', '')

out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

in_path = f"/Users/wuhe/private_code/vines-video/__assets__/poses_skeleton_gifs/dance1_corr.mp4"

print("Processing video using t2v zero: ", in_path)
model = Model(device = "cuda", dtype = torch.float16)
prompt = 'a man dancing'
motion_path = in_path
# motion_path = '__assets__/poses_skeleton_gifs/dance2_corr.mp4'
out_path = os.path.join(out_dir, "zeroflicks_" + out_name)
model.process_zeroflicks_pose(motion_path, prompt=prompt, save_path=out_path, guidance_scale=7.5)



import torch
from model import Model
from utils import USE_OUR_METHOD_FLAG
import os
from time import time

USE_OUR_METHOD_FLAG.set(False)

in_dir = '/home/ubuntu/data_cya/output'

out_dir = '/home/ubuntu/t2v-ljy/output_cya'
os.makedirs(out_dir, exist_ok=True)

for video_name in os.listdir(in_dir):

    in_path = os.path.join(in_dir, video_name)

    print("Processing video using t2v zero: ", video_name)
    tik = time()
    model = Model(device = "cuda", dtype = torch.float16)
    prompt = 'a man dancing'
    motion_path = in_path
    # motion_path = '__assets__/poses_skeleton_gifs/dance2_corr.mp4'
    out_path = os.path.join(out_dir, "baseline_" + video_name)
    model.process_controlnet_pose_orig(motion_path, prompt=prompt, save_path=out_path, guidance_scale=7.5)
    tok = time()
    print("Time elapsed: ", tok - tik)

    print("Processing video using ours pipeline: ", video_name)
    tik = time()
    model = Model(device = "cuda", dtype = torch.float16)
    prompt = 'a man dancing'
    motion_path = in_path
    # motion_path = '__assets__/poses_skeleton_gifs/dance2_corr.mp4'
    out_path = os.path.join(out_dir, "ours_" + video_name)
    model.process_controlnet_pose(motion_path, prompt=prompt, save_path=out_path, guidance_scale=7.5)
    tok = time()
    print("Time elapsed: ", tok - tik)

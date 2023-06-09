import torch
from model import Model
from utils import USE_OUR_METHOD_FLAG
import os
from time import time

USE_OUR_METHOD_FLAG.set(False)

in_prompt = 'a man throwing a ball'
out_name = in_prompt.replace(' ', '_').replace('.', '')

cache_dir = '_motion_cache'
os.makedirs(cache_dir, exist_ok=True)

out_dir = 'output'
os.makedirs(out_dir, exist_ok=True)

# stage 1: generate pose sequence using motion diffusion model. https://github.com/GuyTevet/motion-diffusion-model

os.system("""cd /home/ubuntu/motion-diffusion-model && python -m sample.generate --model_path ./save/humanml_trans_enc_512/model000200000.pt --text_prompt "a man is waving with right hand slowly." --num_repetitions 1""")

# stage 2: generate video using zeroflicks

in_path = f"motion-diffusion-model/save/humanml_trans_enc_512/samples_humanml_trans_enc_512_000200000_seed10_{out_name}/sample00.mp4"

# BUG: ModuleNotFoundError: No module named 'text_to_video_pipeline'
print("Processing video using t2v zero: ", in_path)
tik = time()
model = Model(device = "cuda", dtype = torch.float16)
prompt = 'a man dancing'
motion_path = in_path
# motion_path = '__assets__/poses_skeleton_gifs/dance2_corr.mp4'
out_path = os.path.join(out_dir, "zeroflicks_" + out_name)
model.process_controlnet_pose_orig(motion_path, prompt=prompt, save_path=out_path, guidance_scale=7.5)

import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
from annotator.midas import MidasDetector
import decord

apply_canny = CannyDetector()
apply_openpose = OpenposeDetector()
apply_midas = MidasDetector()

class Flag:
    def __init__(self, value):
        self.value = value
        self.idx = 0
        self.flow = None

    def set(self, value):
        self.value = value

    def val(self):
        return self.value

USE_OUR_METHOD_FLAG = Flag(False)

def add_watermark(image, watermark_path, wm_rel_size=1/16, boundary=5):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    watermark_path = '/home/ubuntu/t2v-ljy/monkey_LOGO.png'
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ('RGBA', 'LA') else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def pre_process_depth(input_video, apply_depth_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_depth_detect:
            detected_map, _ = apply_midas(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def pre_process_pose(input_video, apply_pose_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_pose_detect:
            detected_map, _ = apply_openpose(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def create_gif(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'canny_db.gif')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def prepare_video(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file


class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        global USE_OUR_METHOD_FLAG

        # flow = USE_OUR_METHOD_FLAG.flow
        # USE_OUR_METHOD_FLAG.idx += 1
        # if not is_cross_attention and USE_OUR_METHOD_FLAG.idx < 900:
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)

            # ref_key_s = [key[:, 0], key[:, 0], key[:, 0]]
            # for fl in flow:
            #     prev = ref_key_s[-1] # b d c
            #     prev = rearrange(prev, "b (h w) c -> (b c) h w ", h=prev.shape[1] // 64)
            #     nxt = warp_latents_independently(prev.unsqueeze(0), fl).squeeze(0)
            #     nxt = rearrange(nxt, "(b c) h w -> b (h w) c", b=2)
            #     ref_key_s.append(nxt)
            # key = torch.stack(ref_key_s, dim=1)

            global USE_OUR_METHOD_FLAG
            if USE_OUR_METHOD_FLAG.val():
                key = key.mean(dim=1, keepdim=True).repeat(1, video_length, 1, 1)
            else:
                key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            # ref_val_s = [value[:, 0], value[:, 0], value[:, 0]]
            # for fl in flow:
            #     prev = ref_val_s[-1] # b d c
            #     prev = rearrange(prev, "b (h w) c -> (b c) h w ", h=prev.shape[1] // 64)
            #     nxt = warp_latents_independently(prev.unsqueeze(0), fl).squeeze(0)
            #     nxt = rearrange(nxt, "(b c) h w -> b (h w) c", b=2)
            #     ref_val_s.append(nxt)
            # value = torch.stack(ref_val_s, dim=1)

            if USE_OUR_METHOD_FLAG.val():
                value = value.mean(dim=1, keepdim=True).repeat(1, video_length, 1, 1)
            else:
                value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")
        # else:
        #     if USE_OUR_METHOD_FLAG.idx > 900:
        #         print(f'use normal attention @{USE_OUR_METHOD_FLAG.idx}')

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def pre_process_flow(video_seq):
    """
    get the flow between frames
    """
    video_seq = video_seq.permute(0, 2, 3, 1).cpu().numpy()
    flow = []
    for i in range(len(video_seq) - 1):
        prev_img = cv2.cvtColor(video_seq[i].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        next_img = cv2.cvtColor(video_seq[i + 1].astype(np.uint8), cv2.COLOR_BGR2GRAY)
        flow.append(cv2.calcOpticalFlowFarneback(prev_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0))

        h, w = prev_img.shape[:2]
        y, x = np.mgrid[0:h:10, 0:w:10].reshape(2,-1)
        fx, fy = flow[-1][y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        vis = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        cv2.imwrite(f"./output/flow_{i}.png", vis)
    flow = np.array(flow)
    flow = torch.from_numpy(flow).permute(0, 3, 1, 2).float().cuda()
    return flow

def coords_grid(batch, ht, wd, device):
    # Adapted from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
    coords = torch.meshgrid(torch.arange(
        ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def warp_latents_independently(latents, reference_flow):
    import torchvision.transforms as T
    from torch.nn.functional import grid_sample

    reference_flow = reference_flow.unsqueeze(0).half()
    if len(latents.size()) < 5:
        latents = latents.unsqueeze(0).half()

    _, _, H, W = reference_flow.size() # 2, 512, 512
    b, _, f, h, w = latents.size() # 1, 4, 64, 64

    assert b == 1
    coords0 = coords_grid(f, H, W, device=latents.device).to(latents.dtype)

    coords_t0 = coords0 + reference_flow
    coords_t0[:, 0] /= W
    coords_t0[:, 1] /= H

    coords_t0 = coords_t0 * 2.0 - 1.0

    coords_t0 = T.Resize((h, w))(coords_t0)

    coords_t0 = rearrange(coords_t0, 'f c h w -> f h w c')

    latents_0 = rearrange(latents[0], 'c f h w -> f  c  h w')
    warped = grid_sample(latents_0, coords_t0, mode='nearest', padding_mode='reflection')

    warped = rearrange(warped, '(b f) c h w -> b c f h w', f=f)
    return warped.squeeze(0)
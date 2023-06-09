import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegFileWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
# import cv2
from textwrap import wrap

from pdb import set_trace as st

from PIL import Image

def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=24, radius=2,
                   vis_mode='default', gt_frames=[]):
    matplotlib.use('Agg')
    
    def init(ax_3d):
        # pass
        ax_3d.set_xlim3d([-radius / 2, radius / 2])
        ax_3d.set_ylim3d([0, radius])
        ax_3d.set_zlim3d([-radius *2 / 3, radius / 3])
        # ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        # print(title)
        # fig.suptitle(title, fontsize=10)
        ax_3d.grid(b=False)
        
    
    def humanml2openpose(humanml_kpt):
        # huamnml: [_b, 22, 3]
        _batch = humanml_kpt.shape[0]
        _mapper = { # openpose: humanml
            0: [15],
            1: [16, 17], 
            2: [17], 
            3: [19], 
            4: [21], 
            5: [16], 
            6: [18],
            7: [20],
            8: [2],
            9: [5],
            10: [8],
            11: [1],
            12: [4],
            13: [7],
            14: [15],
            15: [15],
            16: [15],
            17: [15],
        }
        
        
        ret = np.zeros((_batch, 18, 3))
        for _b in range(_batch):
            _huamnml_kpt = humanml_kpt[_b]
            _openpose_kpt = np.zeros((18, 3))
            for _op_idx, _hmml_idxs in _mapper.items():
                _openpose_kpt[_op_idx] = np.mean(_huamnml_kpt[_hmml_idxs], axis=0)  
            ret[_b] = _openpose_kpt
        return ret


    data = joints.copy().reshape(len(joints), -1, 3)
    data *= 1.3  # scale for visualization
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
    colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
    
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
    _limbSeq = [[i[0] - 1, i[1] - 1] for i in limbSeq]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],  [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    colors = np.array(colors) / 255.0
    _openpose = humanml2openpose(data)
    
    def fig2img(fig):
        # convert a Matplotlib figure to a PIL Image and return it
        import io
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    
    out_imgs = []
    for index in range(data.shape[0]):
        # pose = data[index]
        print(f"Processing {index}")
        # create a new figure
        plt.cla()
        plt.clf()
        fig = plt.figure(figsize=figsize)
        plt.tight_layout()
        ax_3d = p3.Axes3D(fig)
        
        fig.add_axes(ax_3d)
        init(ax_3d)
        
        ax_3d.view_init(elev=120, azim=-90)
        ax_3d.dist = 7.5
        # ax_3d.set_box_aspect((1, 1, 1), zoom=7.5)
        
        
        for i in range(18):
            ax_3d.scatter(_openpose[index, i, 0], _openpose[index, i, 1], _openpose[index, i, 2], color=colors[i], s=8)
            
        for i in range(17):
            _index = _limbSeq[i]
            
            # plot ellipses
            linewidth = 4.0
            ax_3d.plot(_openpose[index, _index, 0], _openpose[index, _index, 1], zs=_openpose[index, _index, 2], linewidth=linewidth,
                      color=colors[i])
        # change background to black
        ax_3d.set_facecolor((0, 0, 0))
    
        plt.axis('off')
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])
        
        # convert to pil image
        fig.savefig(f"{save_path}_{index}_mpl.png")
        out_img = fig2img(fig)
        
        out_img.save(f"{save_path}_{index}.png")
        # st()
        out_imgs.append(out_img)
        
    
    # save as video
    import imageio
    imageio.mimsave(save_path, out_imgs, fps=fps)
    
        
# The Original code 

# def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
#                    vis_mode='default', gt_frames=[]):
#     matplotlib.use('Agg')

#     title = '\n'.join(wrap(title, 20))

#     def init():
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_ylim3d([0, radius])
#         ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
#         # print(title)
#         fig.suptitle(title, fontsize=10)
#         ax.grid(b=False)

#     def plot_xzPlane(minx, maxx, miny, minz, maxz):
#         ## Plot a plane XZ
#         verts = [
#             [minx, miny, minz],
#             [minx, miny, maxz],
#             [maxx, miny, maxz],
#             [maxx, miny, minz]
#         ]
#         xz_plane = Poly3DCollection([verts])
#         xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#         ax.add_collection3d(xz_plane)

#     #         return ax

#     # (seq_len, joints_num, 3)
#     data = joints.copy().reshape(len(joints), -1, 3)

#     # preparation related to specific datasets
#     if dataset == 'kit':
#         data *= 0.003  # scale for visualization
#     elif dataset == 'humanml':
#         data *= 1.3  # scale for visualization
#     elif dataset in ['humanact12', 'uestc']:
#         data *= -1.5 # reverse axes, scale for visualization

#     fig = plt.figure(figsize=figsize)
#     plt.tight_layout()
#     ax = p3.Axes3D(fig)
#     init()
#     MINS = data.min(axis=0).min(axis=0)
#     MAXS = data.max(axis=0).max(axis=0)
#     colors_blue = ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"]  # GT color
#     colors_orange = ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"]  # Generation color
#     colors = colors_orange
#     if vis_mode == 'upper_body':  # lower body taken fixed to input motion
#         colors[0] = colors_blue[0]
#         colors[1] = colors_blue[1]
#     elif vis_mode == 'gt':
#         colors = colors_blue

#     frame_number = data.shape[0]
#     #     print(dataset.shape)

#     height_offset = MINS[1]
#     data[:, :, 1] -= height_offset
#     trajec = data[:, 0, [0, 2]]

#     data[..., 0] -= data[:, 0:1, 0]
#     data[..., 2] -= data[:, 0:1, 2]

#     #     print(trajec.shape)

#     def update(index):
#         #         print(index)
#         ax.lines = []
#         ax.collections = []
#         ax.view_init(elev=120, azim=-90)
#         ax.dist = 7.5
#         #         ax =
#         plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
#                      MAXS[2] - trajec[index, 1])
#         #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

#         # if index > 1:
#         #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
#         #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
#         #               color='blue')
#         # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

#         used_colors = colors_blue if index in gt_frames else colors
#         for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
#             if i < 5:
#                 linewidth = 4.0
#             else:
#                 linewidth = 2.0
#             ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
#                       color=color)
#         #         print(trajec[:index, 0].shape)

#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])

#     ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

#     # writer = FFMpegFileWriter(fps=fps)
#     ani.save(save_path, fps=fps)
#     # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
#     # ani.save(save_path, writer='pillow', fps=1000 / fps)

#     plt.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGN Preprocessing: Run CNN → generate probmap → build graph → save everything
"""
import os
import argparse
import numpy as np
import skimage.io
import networkx as nx
import pickle as pkl
import tensorflow as tf
from pathlib import Path

tf.compat.v1.disable_eager_execution()

from model import vessel_segm_vgn
from config import cfg
import util

# Reuse các hàm từ script gốc
def load_image_gray(path):
    im = skimage.io.imread(path, as_gray=True)
    im = im.astype(np.float32)
    if im.max() > 1.0:
        im = im / 255.0
    return im

def build_default_args():
    args = argparse.Namespace()

    # dataset
    args.dataset = "STARE"

    # multiprocessing / graph
    args.use_multiprocessing = False
    args.multiprocessing_num_proc = 1
    args.win_size = 4
    args.edge_type = "srns_geo_dist_binary"
    args.edge_geo_dist_thresh = 10

    # pretrained / save paths (not critical for inference)
    args.pretrained_model = None
    args.save_root = ""

    # cnn module
    args.cnn_model = "driu"
    args.cnn_loss_on = True

    # gnn module
    args.gnn_loss_on = True
    args.gnn_loss_weight = 1.0
    args.gnn_feat_dropout_prob = 0.0
    args.gnn_att_dropout_prob = 0.0
    args.gat_n_heads = [4, 4]
    args.gat_hid_units = [16]
    args.gat_use_residual = False

    # inference module
    args.norm_type = None
    args.use_enc_layer = False
    args.infer_module_loss_masking_thresh = 0.05
    args.infer_module_kernel_size = 3
    args.infer_module_grad_weight = 1.0
    args.infer_module_dropout_prob = 0.0
    args.use_fov_mask = True

    # training (dummy)
    args.do_simul_training = True
    args.max_iters = 1
    args.old_net_ft_lr = 1e-2
    args.new_net_lr = 1e-2
    args.opt = 'adam'
    args.lr_scheduling = 'pc'
    args.lr_decay_tp = 1.
    args.use_graph_update = False
    args.graph_update_period = 10000

    return args

def build_graph_from_probmap(probmap, win_size=4, geo_thresh=10):
    H, W = probmap.shape
    y_quan = sorted(list(set(range(0, H, win_size)) | set([H])))
    x_quan = sorted(list(set(range(0, W, win_size)) | set([W])))

    max_pos = []
    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = probmap[y_quan[y_idx]:y_quan[y_idx+1], x_quan[x_idx]:x_quan[x_idx+1]]
            if np.sum(cur_patch) == 0:
                max_pos.append((y_quan[y_idx] + cur_patch.shape[0]/2, x_quan[x_idx] + cur_patch.shape[1]/2))
            else:
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx] + temp[0], x_quan[x_idx] + temp[1]))

    G = nx.Graph()
    for node_idx, (ny, nx_) in enumerate(max_pos):
        G.add_node(node_idx, y=int(round(ny)), x=int(round(nx_)))

    vesselness = probmap
    im_y, im_x = vesselness.shape
    node_list = list(G.nodes)
    for i, n in enumerate(node_list):
        phi = np.ones_like(vesselness)
        ny = int(G.nodes[n]['y'])
        nx_ = int(G.nodes[n]['x'])
        if ny < 0 or ny >= im_y or nx_ < 0 or nx_ >= im_x:
            continue
        phi[ny, nx_] = -1
        if vesselness[ny, nx_] == 0:
            continue

        # travel time (skfmm)
        try:
            tt = skfmm.travel_time(phi, vesselness, narrow=geo_thresh)
        except Exception:
            # fallback: use euclidean distance if skfmm fails
            tt = np.full_like(vesselness, np.inf, dtype=float)
            for j in node_list[i+1:]:
                yy = int(G.nodes[j]['y'])
                xx = int(G.nodes[j]['x'])
                tt[yy, xx] = np.sqrt((yy - ny)**2 + (xx - nx_)**2)

        for n_comp in node_list[i+1:]:
            yy = int(G.nodes[n_comp]['y'])
            xx = int(G.nodes[n_comp]['x'])
            geo_dist = tt[yy, xx]
            if geo_dist < geo_thresh:
                # weight same as train_VGN
                G.add_edge(n, n_comp, weight=geo_thresh/(geo_thresh + geo_dist))
    return G


def robust_restore(sess, saver, ckpt_path):
    try:
        saver.restore(sess, ckpt_path)
        print("[INFO] Full checkpoint restore succeeded.")
        return True
    except Exception as e:
        print("[WARN] Full restore failed:", str(e))
        print("[INFO] Attempting partial restore (restore only variables present in checkpoint)...")

    # get variables in checkpoint
    try:
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        ckpt_vars = set(reader.get_variable_to_shape_map().keys())
    except Exception as e2:
        print("[ERROR] Could not read checkpoint variables:", e2)
        return False

    # build var_list of variables that exist in the checkpoint
    var_list = {}
    for v in tf.compat.v1.global_variables():
        v_name = v.name.split(':')[0]
        if v_name in ckpt_vars:
            var_list[v_name] = v

    if len(var_list) == 0:
        print("[ERROR] No matching variables found between graph and checkpoint.")
        return False

    print("[INFO] Restoring %d variables from checkpoint (partial restore)..." % len(var_list))
    partial_saver = tf.compat.v1.train.Saver(var_list=var_list, write_version=tf.compat.v1.train.SaverDef.V2)
    try:
        partial_saver.restore(sess, ckpt_path)
        print("[INFO] Partial restore succeeded.")
        return True
    except Exception as e3:
        print("[ERROR] Partial restore failed:", e3)
        return False


class VGNPreprocessor:
    def __init__(self, model_path, win_size=4, geo_thresh=10.0):
        self.model_path = model_path
        self.win_size = win_size
        self.geo_thresh = geo_thresh

        # Build model
        args = build_default_args()
        args.win_size = win_size
        args.edge_geo_dist_thresh = geo_thresh
        self.network = vessel_segm_vgn(args, None)

        # TF session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.InteractiveSession(config=config)

        saver = tf.compat.v1.train.Saver(max_to_keep=100)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        print("[Preprocessor] Loading model:", model_path)
        if not robust_restore(self.sess, saver, model_path):
            raise RuntimeError("Cannot load checkpoint!")

    def process(self, img_path, output_dir):
        img_path = Path(img_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem

        # Load image
        img = load_image_gray(img_path)
        H, W = img.shape
        img_rgb = np.stack([img, img, img], axis=-1)
        img_batch = img_rgb[None, ...].astype(np.float32)  # (1,H,W,3)
        dummy_mask = np.ones((1, H, W, 1), dtype=np.float32)

        # === Step 1: Run CNN để lấy probmap ===
        print(f"[{stem}] Running CNN for probmap...")
        feed = {
            self.network.imgs: img_batch,
            self.network.is_lr_flipped: False,
            self.network.is_ud_flipped: False,
            self.network.rot90_num: 0,
            self.network.fov_masks: dummy_mask,
        }
        try:
            probmap = self.sess.run(self.network.img_fg_prob, feed_dict=feed)
        except:
            probmap = self.sess.run(self.network.post_cnn_img_fg_prob, feed_dict=feed)
        probmap = probmap[0, :, :, 0]  # (H,W)

        # === Step 2: Build graph ===
        print(f"[{stem}] Building graph (win={self.win_size}, geo={self.geo_thresh})...")
        G = build_graph_from_probmap(probmap, win_size=self.win_size, geo_thresh=self.geo_thresh)

        # === Lưu tất cả ===
        save_dict = {
            'probmap': probmap.astype(np.float32),
            'graph': G,
            'win_size': self.win_size,
            'geo_thresh': self.geo_thresh,
            'image_shape': (H, W)
        }
        pkl_path = output_dir / f"{stem}_vgn_preproc.pkl"
        np.savez_compressed(output_dir / f"{stem}_probmap.npz", probmap=probmap)
        with open(pkl_path, 'wb') as f:
            pkl.dump(save_dict, f)

        print(f"[{stem}] Saved to {output_dir}")
        return pkl_path

    def close(self):
        self.sess.close()


# ==========================
# CLI
# ==========================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to image or folder')
    parser.add_argument('--model', required=True, help='VGN checkpoint')
    parser.add_argument('--output', default='vgn_preproc', help='Output folder')
    parser.add_argument('--win_size', type=int, default=4)
    parser.add_argument('--geo_thresh', type=float, default=10.0)
    args = parser.parse_args()

    preprocessor = VGNPreprocessor(args.model, args.win_size, args.geo_thresh)

    if os.path.isdir(args.img):
        for impath in Path(args.img).glob("*.png"):
            try:
                preprocessor.process(impath, args.output)
            except Exception as e:
                print(f"[ERROR] {impath}: {e}")
    else:
        preprocessor.process(args.img, args.output)

    preprocessor.close()
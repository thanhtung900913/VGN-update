#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import numpy as np
import skimage.io
import pickle as pkl
import tensorflow as tf
from pathlib import Path
import networkx as nx

# Tắt chế độ Eager Execution để chạy Graph mode (TF 1.x style)
tf.compat.v1.disable_eager_execution()

from model import vessel_segm_vgn
from config import cfg
import util


def load_image_gray(path):
    """Load ảnh và chuẩn hóa về grayscale [0, 1]"""
    im = skimage.io.imread(path, as_gray=True).astype(np.float32)
    if im.max() > 1.0:
        im /= 255.0
    return im


def pad_to_square_stride(img, stride=32):
    """
    Đệm ảnh thành hình VUÔNG (Square) sao cho cạnh chia hết cho stride.
    Lý do: Khi win_size thay đổi, một số module GNN có xu hướng yêu cầu
    hoặc tạo ra feature map hình vuông dựa trên cạnh lớn nhất.
    """
    h, w = img.shape[:2]
    
    # Tìm cạnh lớn nhất
    max_dim = max(h, w)
    
    # Tính toán kích thước mới là bội số của stride
    # Ví dụ: max(605, 700) = 700 -> lên 704 (chia hết 32)
    new_size = int(math.ceil(max_dim / stride) * stride)
    
    pad_h = new_size - h
    pad_w = new_size - w
    
    # Pad (top, bottom), (left, right)
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        img_padded = img
        
    return img_padded, (h, w)


def build_default_args():
    """Thiết lập các tham số mặc định cho mô hình VGN"""
    args = argparse.Namespace()
    args.dataset = "STARE"
    args.use_multiprocessing = False
    args.multiprocessing_num_proc = 1
    
    # Cập nhật win_size theo yêu cầu của user
    args.win_size = 8 
    
    args.edge_type = "srns_geo_dist_binary"
    args.edge_geo_dist_thresh = 10
    args.pretrained_model = None
    args.save_root = ""
    args.cnn_model = "driu"
    args.cnn_loss_on = True
    args.gnn_loss_on = True
    args.gnn_loss_weight = 1.0
    args.gnn_feat_dropout_prob = 0.0
    args.gnn_att_dropout_prob = 0.0
    args.gat_n_heads = [4, 4]
    args.gat_hid_units = [16]
    args.gat_use_residual = False
    args.norm_type = None
    args.use_enc_layer = False
    args.infer_module_loss_masking_thresh = 0.05
    args.infer_module_kernel_size = 3
    args.infer_module_grad_weight = 1.0
    args.infer_module_dropout_prob = 0.0
    args.use_fov_mask = True
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


def robust_restore(sess, saver, ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path) and not os.path.exists(ckpt_path + ".index"):
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        return False

    try:
        saver.restore(sess, ckpt_path)
        print(f"[INFO] Full checkpoint restore succeeded.")
        return True
    except Exception as e:
        print(f"[WARN] Full restore failed: {e}")
        print("[INFO] Attempting partial restore...")

    try:
        reader = tf.compat.v1.train.NewCheckpointReader(ckpt_path)
        ckpt_vars = set(reader.get_variable_to_shape_map().keys())
    except Exception as e2:
        print(f"[ERROR] Could not read checkpoint variables: {e2}")
        return False

    var_list = {}
    for v in tf.compat.v1.global_variables():
        v_name = v.name.split(':')[0]
        if v_name in ckpt_vars:
            var_list[v_name] = v

    if len(var_list) == 0:
        print("[ERROR] No matching variables found.")
        return False

    print(f"[INFO] Restoring {len(var_list)} variables from checkpoint (partial restore)...")
    partial_saver = tf.compat.v1.train.Saver(
        var_list=var_list, 
        write_version=tf.compat.v1.train.SaverDef.V2
    )
    
    try:
        partial_saver.restore(sess, ckpt_path)
        print("[INFO] Partial restore succeeded.")
        return True
    except Exception as e3:
        print(f"[ERROR] Partial restore failed: {e3}")
        return False


class VGNInferencer:
    def __init__(self, model_path):
        self.args = build_default_args()
        self.network = vessel_segm_vgn(self.args, None)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.InteractiveSession(config=config)
        
        saver = tf.compat.v1.train.Saver(max_to_keep=100)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        print("[Inferencer] Loading model:", model_path)
        if not robust_restore(self.sess, saver, model_path):
            raise RuntimeError(f"Failed to restore model from {model_path}")

    def infer(self, prob_path, graph_path, img_path=None, output_path=None):
        # Load probmap
        probmap = skimage.io.imread(prob_path).astype(np.float32)
        if probmap.max() > 1:
            probmap /= 255.0
        
        # Lấy kích thước ảnh gốc
        orig_h, orig_w = probmap.shape
        print(f"[INFO] Original Image Size: H={orig_h}, W={orig_w}")

        # --- PAD PROBMAP ---
        probmap_padded, (real_h, real_w) = pad_to_square_stride(probmap, stride=32)
        pad_h, pad_w = probmap_padded.shape
        print(f"[INFO] Padded Image Size: H={pad_h}, W={pad_w} (Square, Stride 32)")

        # Load graph
        with open(graph_path, "rb") as f:
            data = pkl.load(f)
        G = data["graph"] if isinstance(data, dict) and "graph" in data else data

        # Load image (optional)
        if img_path:
            img = load_image_gray(img_path)
            # Resize img nếu kích thước khác probmap
            if img.shape != (orig_h, orig_w):
                 img = skimage.transform.resize(img, (orig_h, orig_w))
            
            # Pad ảnh gốc giống probmap
            img_padded, _ = pad_to_square_stride(img, stride=32)
        else:
            img_padded = probmap_padded
        
        # --- [NEW] DATA NORMALIZATION ---
        # Chuẩn hóa ảnh về (mean=0, std=1) trước khi đưa vào mạng
        # Đây là bước quan trọng để tránh hiện tượng "ảnh xám" (output saturation)
        mean_val = np.mean(img_padded)
        std_val = np.std(img_padded)
        if std_val > 1e-5:
            img_normalized = (img_padded - mean_val) / std_val
        else:
            img_normalized = img_padded - mean_val
        # --------------------------------

        # Chuẩn bị input (dùng ảnh đã Normalized)
        img_rgb = np.stack([img_normalized, img_normalized, img_normalized], axis=-1)
        img_batch = img_rgb[None, ...].astype(np.float32)
        
        # Mask
        dummy_mask = np.ones((1, pad_h, pad_w, 1), np.float32)

        # Prepare graph data
        num_nodes = len(G.nodes)
        node_byxs = util.get_node_byx_from_graph(G, [num_nodes])
        
        # --- CLIP COORDINATES ---
        max_node_y = np.max(node_byxs[:, 1])
        max_node_x = np.max(node_byxs[:, 2])
        if max_node_y >= pad_h or max_node_x >= pad_w:
            print(f"[WARN] Graph nodes out of bounds! Clipping to ({pad_h}, {pad_w})...")
            node_byxs[:, 1] = np.clip(node_byxs[:, 1], 0, pad_h - 1)
            node_byxs[:, 2] = np.clip(node_byxs[:, 2], 0, pad_w - 1)

        adj = nx.adjacency_matrix(G, weight=None).astype(float)
        adj_norm = util.preprocess_graph_gat(adj)

        feed = {
            self.network.imgs: img_batch,
            self.network.node_byxs: node_byxs,
            self.network.adj: adj_norm,
            self.network.fov_masks: dummy_mask,
            self.network.pixel_weights: dummy_mask,
            self.network.is_lr_flipped: False,
            self.network.is_ud_flipped: False,
            self.network.rot90_num: 0,
            self.network.gnn_feat_dropout: 0.0,
            self.network.gnn_att_dropout: 0.0,
            self.network.post_cnn_dropout: 0.0,
        }

        print(f"[INFO] Running inference...")
        out_prob = self.sess.run(self.network.post_cnn_img_fg_prob, feed_dict=feed)
        
        # Lấy kết quả
        out_prob = out_prob[0, :, :, 0] # Shape: (pad_h, pad_w)

        # --- CROP BACK TO ORIGINAL SIZE ---
        out_prob = out_prob[:orig_h, :orig_w]
        
        # In thông tin thống kê của output để debug
        p_min, p_max, p_mean = out_prob.min(), out_prob.max(), out_prob.mean()
        print(f"[DEBUG] Output Stats: Min={p_min:.4f}, Max={p_max:.4f}, Mean={p_mean:.4f}")
        
        if p_max - p_min < 0.01:
            print("[WARN] Output probability map has very low contrast (Gray image).")
            print("       Possible causes: Input normalization issue, incorrect pretrained model, or wrong input image.")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            skimage.io.imsave(output_path, (out_prob * 255).astype(np.uint8))
            print(f"[SUCCESS] Saved result to: {output_path}")

        return out_prob

    def close(self):
        self.sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGN Inference Script (Square Padding + Norm)")
    parser.add_argument("--prob", required=True, help="Path to *_prob.png")
    parser.add_argument("--graph", required=True, help="Path to *.graph_res / *.pkl")
    parser.add_argument("--img", default=None, help="Original image (optional)")
    parser.add_argument("--model", required=True, help="Path to VGN checkpoint")
    parser.add_argument("--output", default="vgn_final.png", help="Output path")
    args = parser.parse_args()

    try:
        infer = VGNInferencer(args.model)
        infer.infer(args.prob, args.graph, args.img, args.output)
        infer.close()
    except Exception as e:
        print(f"\n[CRITICAL ERROR]: {e}")
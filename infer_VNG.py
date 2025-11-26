#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full VGN single-image inference:
 - build model exactly like train_VGN
 - load VGN checkpoint (with robust fallback to partial restore)
 - run CNN to get probmap
 - build graph from probmap (local maxima + skfmm geodesic)
 - run GNN + inference module and save final segmentation map
"""
import os
import argparse
import numpy as np
import skimage.io
import networkx as nx
import skfmm
import pickle as pkl
import tensorflow as tf

# disable eager for TF1-style code
tf.compat.v1.disable_eager_execution()

# import project modules (assumes you're running from repo root)
from model import vessel_segm_vgn
from config import cfg
import util

# -----------------------
# Helper: build args like train_VGN.parse_args()
# -----------------------
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

# -----------------------
# Helper: load grayscale image [0..1]
# -----------------------
def load_image_gray(path):
    im = skimage.io.imread(path, as_gray=True)
    im = im.astype(np.float32)
    if im.max() > 1.0:
        im = im / 255.0
    return im

# -----------------------
# Build graph from probmap (same logic as train -> make_train_qual_res)
# win_size and geo_thresh should match args.win_size / args.edge_geo_dist_thresh
# -----------------------
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

# -----------------------
# Robust restore: try full saver.restore; on failure, restore variable-subset by intersection
# -----------------------
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

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='Path to input image (grayscale)')
    parser.add_argument('--model', required=True, help='Path to VGN checkpoint prefix (e.g. models/.../VGN_STARE.ckpt)')
    parser.add_argument('--output', default='vgn_out.png', help='Output segmentation map (png)')
    parser.add_argument('--win_size', type=int, default=4, help='win_size used for graph node sampling')
    parser.add_argument('--geo_thresh', type=float, default=10.0, help='geodesic threshold for edges')
    args = parser.parse_args()

    # load image
    img = load_image_gray(args.img)
    H, W = img.shape
    img_rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)
    img_batch = img_rgb.reshape(1, H, W, 3)  # (1, H, W, 3)

    # [FIX] Tạo mask 1 kênh (1 channel) thay vì dùng ones_like trên img_batch (3 channels)
    # Tensor fov_masks yêu cầu shape (Batch, H, W, 1)
    dummy_mask = np.ones((1, H, W, 1), dtype=np.float32)

    # build args and network like in train_VGN
    vgn_args = build_default_args()
    vgn_args.win_size = args.win_size
    vgn_args.edge_geo_dist_thresh = args.geo_thresh

    print("[INFO] Building VGN model (this may create training ops too) ...")
    network = vessel_segm_vgn(vgn_args, None)

    # session config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession(config=config)

    # init and restore
    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    sess.run(tf.compat.v1.global_variables_initializer())

    print("[INFO] Loading checkpoint:", args.model)
    ok = robust_restore(sess, saver, args.model)
    if not ok:
        print("[ERROR] Could not restore checkpoint. Abort.")
        return

    # Step1: run CNN to get initial probmap
    print("[INFO] Running CNN to get initial vesselness probmap...")
    feed_cnn = {
        network.imgs: img_batch,
        network.is_lr_flipped: False,
        network.is_ud_flipped: False,
        network.rot90_num: 0,
        network.fov_masks: dummy_mask # [FIX] Sử dụng dummy_mask 1 kênh
    }

    try:
        cnn_prob = sess.run(network.img_fg_prob, feed_dict=feed_cnn)
    except Exception as e:
        # some models might use different tensor name; try post_cnn_img_fg_prob as fallback
        print("[WARN] Running network.img_fg_prob failed:", e)
        cnn_prob = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed_cnn)

    cnn_prob = cnn_prob.reshape(H, W)

    # Step2: build graph from probmap
    print("[INFO] Building graph from probmap (win_size=%d, geo_thresh=%s) ..." % (args.win_size, str(args.geo_thresh)))
    G = build_graph_from_probmap(cnn_prob, win_size=args.win_size, geo_thresh=args.geo_thresh)

    # build node_byx and adjacency normalization (same util used in train)
    num_nodes = len(G.nodes)
    node_byxs = util.get_node_byx_from_graph(G, [num_nodes])

    if 'geo_dist_weighted' in vgn_args.edge_type:
        adj = nx.adjacency_matrix(G)
    else:
        adj = nx.adjacency_matrix(G, weight=None).astype(float)
    adj_norm = util.preprocess_graph_gat(adj)

    # Step3: run full inference (GNN + inference module)
    print("[INFO] Running full VGN inference (GNN + fusion)...")
    feed_full = {
        network.imgs: img_batch,
        network.node_byxs: node_byxs,
        network.adj: adj_norm,
        network.fov_masks: dummy_mask,      # [FIX] Sử dụng dummy_mask 1 kênh
        network.pixel_weights: dummy_mask,  # [FIX] Sử dụng dummy_mask 1 kênh cho weights
        network.is_lr_flipped: False,
        network.is_ud_flipped: False,
        network.rot90_num: 0,
        network.gnn_feat_dropout: 0.0,
        network.gnn_att_dropout: 0.0,
        network.post_cnn_dropout: 0.0,
    }

    out_prob = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed_full)
    out_prob = out_prob.reshape(H, W)
    out_img = (out_prob * 255.0).astype(np.uint8)

    skimage.io.imsave(args.output, out_img)
    print("[INFO] Saved output:", args.output)

    sess.close()

if __name__ == '__main__':
    main()
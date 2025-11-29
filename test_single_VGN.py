# test_VGN_single.py
# Inference VGN cho 1 ảnh duy nhất
# Author: ChatGPT (2025)

import argparse
import numpy as np
import skimage.io
import networkx as nx
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import util
from model import vessel_segm_vgn


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGN on a single image")
    parser.add_argument("--img", required=True, help="Path to raw RGB image")
    parser.add_argument("--prob", required=True, help="Path to CNN prob map (0-1)")
    parser.add_argument("--graph", required=True, help="Graph file .pkl")
    parser.add_argument("--model", required=True, help="Path to VGN .ckpt")
    parser.add_argument("--out", required=True, help="Output PNG file")
    parser.add_argument("--win_size", default=16, type=int)
    parser.add_argument("--edge_geo_dist_thresh", default=40, type=float)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load image (raw RGB)
    img = skimage.io.imread(args.img).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w, _ = img.shape
    img_batch = img[np.newaxis, ...]

    # Fake label (không dùng)
    label_batch = np.zeros((1, h, w, 1), dtype=np.float32)

    # Load CNN probability map
    prob_map = skimage.io.imread(args.prob).astype(np.float32)
    if prob_map.max() > 1.0:
        prob_map /= 255.0
    prob_map = prob_map.reshape(h, w)

    # Load graph
    graph = nx.read_gpickle(args.graph)
    graph = nx.convert_node_labels_to_integers(graph)

    # Convert graph → node_byxs + adj matrix
    node_byxs = util.get_node_byx_from_graph(graph, [graph.number_of_nodes()])
    adj = nx.adjacency_matrix(graph, weight=None).astype(float)
    adj_norm = util.preprocess_graph_gat(adj)

    # Build model
    print("Building model...")
    network = vessel_segm_vgn(args, None)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    print("Loading model checkpoint...")
    saver.restore(sess, args.model)

    # Forward pass
    feed = {
        network.imgs: img_batch,
        network.conv_feats: np.zeros((1, h, w, 16), dtype=np.float32),  # dummy
        network.node_byxs: node_byxs,
        network.adj: adj_norm,
        network.is_lr_flipped: False,
        network.is_ud_flipped: False,
    }

    # CNN features yêu cầu đầy đủ keys → tạo dummy
    for k in network.cnn_feat.keys():
        feed[network.cnn_feat[k]] = np.zeros(
            (1,) + tuple(network.cnn_feat[k].shape.as_list()[1:]), np.float32
        )
    for k in network.cnn_feat_spatial_sizes.keys():
        feed[network.cnn_feat_spatial_sizes[k]] = network.cnn_feat_spatial_sizes[k]

    print("Running inference...")
    res = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed)
    res = res[0, :, :, 0]

    # Save
    out_img = (res * 255).astype(np.uint8)
    skimage.io.imsave(args.out, out_img)

    print("Saved result to:", args.out)
    sess.close()


if __name__ == "__main__":
    main()

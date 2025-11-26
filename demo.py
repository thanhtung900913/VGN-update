import tensorflow as tf
import numpy as np
import skimage.io
import skimage.transform
import networkx as nx
import skfmm
import os
import util
from model import vessel_segm_vgn
from config import cfg


def make_graph(fg_prob_map, win_size, edge_geo_dist_thresh):
    im_y, im_x = fg_prob_map.shape
    y_quan = list(range(0, im_y, win_size)) + [im_y]
    x_quan = list(range(0, im_x, win_size)) + [im_x]

    max_pos = []
    for yi in range(len(y_quan) - 1):
        for xi in range(len(x_quan) - 1):
            patch = fg_prob_map[y_quan[yi]:y_quan[yi+1], x_quan[xi]:x_quan[xi+1]]
            if patch.sum() == 0:
                max_pos.append((y_quan[yi] + patch.shape[0]//2,
                                x_quan[xi] + patch.shape[1]//2))
            else:
                pos = np.unravel_index(np.argmax(patch), patch.shape)
                max_pos.append((y_quan[yi] + pos[0], x_quan[xi] + pos[1]))

    graph = nx.Graph()
    for idx, (y, x) in enumerate(max_pos):
        graph.add_node(idx, y=y, x=x)

    for i in range(len(max_pos)):
        y1, x1 = max_pos[i]
        phi = np.ones_like(fg_prob_map)
        phi[y1, x1] = -1

        tt = skfmm.travel_time(phi, fg_prob_map, narrow=edge_geo_dist_thresh)

        for j in range(i+1, len(max_pos)):
            y2, x2 = max_pos[j]
            dist = tt[y2, x2]
            if dist < edge_geo_dist_thresh:
                graph.add_edge(i, j, weight=1.0)

    return graph


def run_single_image(image_path, model_path,
                     win_size=16, edge_thresh=40):

    # Load image
    img = skimage.io.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    img = skimage.transform.resize(img, (512, 512), preserve_range=True)

    img_inp = img.astype(np.float32)[None, :, :, :]

    # Build network
    args = type('', (), {})()  # dummy args
    args.dataset = "DEMO"
    args.cnn_loss_on = False
    args.gnn_loss_on = False
    args.gat_n_heads = [4, 4]
    args.gat_hid_units = [16]
    args.gat_use_residual = False

    network = vessel_segm_vgn(args, None)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)

    # CNN inference
    fg_prob_map = sess.run(
        network.img_fg_prob,
        feed_dict={network.imgs: img_inp, network.labels: np.zeros_like(img_inp)}
    )[0, :, :, 0]

    # Build graph
    graph = make_graph(fg_prob_map, win_size, edge_thresh)

    # Prepare GNN input
    graph = nx.convert_node_labels_to_integers(graph)
    node_byxs = util.get_node_byx_from_graph(graph, [graph.number_of_nodes()])

    adj = nx.adjacency_matrix(graph, weight=None).astype(float)
    adj_norm = util.preprocess_graph_gat(adj)

    feed_dict = {
        network.imgs: img_inp,
        network.conv_feats: sess.run(network.conv_feats,
                                     feed_dict={network.imgs: img_inp,
                                                network.labels: np.zeros_like(img_inp)}),
        network.node_byxs: node_byxs,
        network.adj: adj_norm,
        network.is_lr_flipped: False,
        network.is_ud_flipped: False
    }

    result = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed_dict)
    result = result.reshape((fg_prob_map.shape[0], fg_prob_map.shape[1]))

    sess.close()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--model", required=True, help="Model .ckpt path")
    parser.add_argument("--output", default="result.png")
    args = parser.parse_args()

    res = run_single_image(args.image, args.model)
    skimage.io.imsave(args.output, (res * 255).astype(np.uint8))

    print("Saved result to", args.output)

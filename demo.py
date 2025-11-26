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
    """
    Thực hiện suy luận VGN trên một ảnh đơn
    
    Args:
        image_path: Đường dẫn tới ảnh input
        model_path: Đường dẫn tới checkpoint model (không cần .ckpt extension)
        win_size: Kích thước cửa sổ cho graph construction
        edge_thresh: Ngưỡng khoảng cách hình học cho graph edges
    
    Returns:
        result: Probability map kết quả (giá trị từ 0 đến 1)
    """

    # Load image
    print(f"Loading image from {image_path}...")
    img = skimage.io.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    
    # Resize để tránh lỗi kích thước
    img = skimage.transform.resize(img, (512, 512), preserve_range=True)
    img_inp = img.astype(np.float32)[None, :, :, :]
    
    print("Image loaded and preprocessed")

    # Build network
    print("Building VGN model...")
    args = type('', (), {})()  # dummy args
    args.dataset = "DEMO"
    args.cnn_model = "driu"  # CNN backbone model
    args.cnn_loss_on = False
    args.gnn_loss_on = False
    args.gat_n_heads = [4, 4]
    args.gat_hid_units = [16]
    args.gat_use_residual = False
    args.win_size = win_size
    args.norm_type = 'GN'
    args.infer_module_kernel_size = 3
    args.use_enc_layer = False

    network = vessel_segm_vgn(args, None)
    print("Model built successfully")

    # Create TensorFlow session and restore checkpoint
    print(f"Loading model checkpoint from {model_path}...")
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    print("Checkpoint restored")

    # CNN inference - lấy probability map từ CNN
    print("Running CNN inference...")
    fg_prob_map = sess.run(
        network.img_fg_prob,
        feed_dict={network.imgs: img_inp, network.labels: np.zeros_like(img_inp)}
    )[0, :, :, 0]

    # Build graph từ probability map
    print(f"Building graph with win_size={win_size}, edge_thresh={edge_thresh}...")
    graph = make_graph(fg_prob_map, win_size, edge_thresh)
    print(f"Graph built with {graph.number_of_nodes()} nodes")

    # Prepare GNN input
    graph = nx.convert_node_labels_to_integers(graph)
    node_byxs = util.get_node_byx_from_graph(graph, [graph.number_of_nodes()])

    adj = nx.adjacency_matrix(graph, weight=None).astype(float)
    adj_norm = util.preprocess_graph_gat(adj)

    # Prepare feed_dict cho GNN inference
    feed_dict = {
        network.imgs: img_inp,
        network.labels: np.zeros_like(img_inp),
        network.conv_feats: sess.run(network.conv_feats,
                                     feed_dict={network.imgs: img_inp,
                                                network.labels: np.zeros_like(img_inp)}),
        network.node_byxs: node_byxs,
        network.adj: adj_norm,
        network.is_lr_flipped: False,
        network.is_ud_flipped: False,
        network.pixel_weights: np.ones_like(img_inp),
        network.post_cnn_dropout: 0.0,
        network.gnn_feat_dropout: 0.0,
        network.gnn_att_dropout: 0.0,
        network.rot90_num: 0
    }

    # Run inference module để lấy kết quả cuối cùng
    print("Running GNN and inference module...")
    result = sess.run(network.post_cnn_img_fg_prob, feed_dict=feed_dict)
    result = result.reshape((fg_prob_map.shape[0], fg_prob_map.shape[1]))

    sess.close()
    print("Inference completed")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", required=True, help="Model checkpoint path (without .ckpt extension)")
    parser.add_argument("--output", default="result.png", help="Output result path")
    parser.add_argument("--win_size", type=int, default=16, help="Window size for graph construction")
    parser.add_argument("--edge_thresh", type=int, default=40, help="Edge threshold for graph construction")
    args = parser.parse_args()

    print("="*60)
    print("VGN (Vessel Graph Network) Inference Demo")
    print("="*60)
    
    res = run_single_image(args.image, args.model, 
                          win_size=args.win_size, 
                          edge_thresh=args.edge_thresh)
    
    # Save result
    skimage.io.imsave(args.output, (res * 255).astype(np.uint8))

    print("="*60)
    print(f"Saved result to {args.output}")
    print("="*60)

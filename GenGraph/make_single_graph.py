import numpy as np
import skimage.io
import os
import networkx as nx
import pickle as pkl
import scipy.sparse as sp
from scipy import ndimage
import mahotas as mh
import multiprocessing
import matplotlib.pyplot as plt
import argparse
import skfmm
from scipy.ndimage.morphology import distance_transform_edt

from tqdm import tqdm

import _init_paths
from bwmorph import bwmorph
from config import cfg
import util
import pickle

DEBUG = False


# ============================
#       PARSE ARGS
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description='Generate graph for ONE image')
    parser.add_argument('--dataset', default='STARE', type=str,
                        help='Dataset: DRIVE / STARE / CHASE_DB1 / HRF')

    parser.add_argument('--input_img', required=True, type=str,
                        help='Path tới file .ppm cần tạo graph')

    parser.add_argument('--use_multiprocessing', default=False, type=bool)
    parser.add_argument('--source_type', default='result', type=str)
    parser.add_argument('--win_size', default=4, type=int)
    parser.add_argument('--edge_method', default='geo_dist', type=str)
    parser.add_argument('--edge_dist_thresh', default=10, type=float)

    return parser.parse_args()


# ============================
#      MAIN GRAPH FUNCTION
# ============================
def generate_graph_using_srns(args_tuple):

    img_name, im_root_path, cnn_result_root_path, params = args_tuple
    win_size_str = '%.2d_%.2d' % (params.win_size, params.edge_dist_thresh)

    if params.source_type == 'gt':
        win_size_str += '_gt'

    # ======================
    # Xác định extension ảnh
    # ======================
    if params.dataset == 'DRIVE':
        im_ext = '_image.tif'
        label_ext = '_label.gif'
        len_y = 592
        len_x = 592

    elif params.dataset == 'STARE':
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
        len_y = 704
        len_x = 704

    elif params.dataset == 'CHASE_DB1':
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        len_y = 1024
        len_x = 1024

    elif params.dataset == 'HRF':
        im_ext = '.bmp'
        label_ext = '.tif'
        len_y = 768
        len_x = 768

    # ==========================
    # Lấy tên file ví dụ: "im0001"
    # ==========================
    filename = os.path.basename(img_name).replace('.ppm', '').replace('.jpg', '').replace('.bmp', '')
    cur_filename = filename

    print(f"\n📌 Processing image: {cur_filename}")

    # ==========================
    # Xác định đường dẫn file GT + prob
    # ==========================
    cur_im_path = img_name  # Ảnh gốc chính là file input bạn truyền vào
    cur_gt_mask_path = os.path.join(im_root_path, cur_filename + label_ext)

    if params.source_type == 'gt':
        cur_res_prob_path = cur_gt_mask_path
    else:
        cur_res_prob_path = os.path.join(cnn_result_root_path, cur_filename + '_prob.png')

    # ==========================
    # File output
    # ==========================
    cur_vis_res_im_savepath = os.path.join(cnn_result_root_path,
        cur_filename + '_' + win_size_str + '_vis_graph_res_on_im.png')

    cur_vis_res_mask_savepath = os.path.join(cnn_result_root_path,
        cur_filename + '_' + win_size_str + '_vis_graph_res_on_mask.png')

    cur_res_graph_savepath = os.path.join(cnn_result_root_path,
        cur_filename + '_' + win_size_str + '.graph_res')

    # ==========================
    # Load ảnh + mask + prob
    # ==========================
    im = skimage.io.imread(cur_im_path)
    gt_mask = skimage.io.imread(cur_gt_mask_path)
    gt_mask = (gt_mask.astype(float) / 255) >= 0.5

    vesselness = skimage.io.imread(cur_res_prob_path).astype(float) / 255

    # pad hình theo kích thước chuẩn dataset
    temp = np.copy(im)
    im = np.zeros((len_y, len_x, 3), dtype=temp.dtype)
    im[:temp.shape[0], :temp.shape[1]] = temp

    temp = np.copy(gt_mask)
    gt_mask = np.zeros((len_y, len_x), dtype=temp.dtype)
    gt_mask[:temp.shape[0], :temp.shape[1]] = temp

    temp = np.copy(vesselness)
    vesselness = np.zeros((len_y, len_x), dtype=temp.dtype)
    vesselness[:temp.shape[0], :temp.shape[1]] = temp

    # ==========================
    # Tìm local maxima
    # ==========================
    im_y, im_x = im.shape[0], im.shape[1]
    y_quan = sorted(list(set(range(0, im_y, params.win_size)) | {im_y}))
    x_quan = sorted(list(set(range(0, im_x, params.win_size)) | {im_x}))

    max_pos = []

    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx+1],
                                   x_quan[x_idx]:x_quan[x_idx+1]]

            if np.sum(cur_patch) == 0:
                cy = y_quan[y_idx] + cur_patch.shape[0] // 2
                cx = x_quan[x_idx] + cur_patch.shape[1] // 2
                max_pos.append((int(cy), int(cx)))
            else:
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx] + temp[0], x_quan[x_idx] + temp[1]))

    # ==========================
    # Tạo graph
    # ==========================
    graph = nx.Graph()

    # ---- Progress bar nodes ----
    for idx, (ny, nx_) in enumerate(
            tqdm(max_pos, desc=f"{cur_filename} - Adding nodes")):
        graph.add_node(idx, kind='MP', y=int(ny), x=int(nx_), label=idx)

    speed = vesselness
    if params.source_type == 'gt':
        speed = bwmorph(speed, 'dilate', n_iter=1).astype(float)

    node_list = list(graph.nodes)
    edge_dist_thresh_sq = params.edge_dist_thresh ** 2

    # ---- Progress bar edges ----
    for i, n in enumerate(
            tqdm(node_list, desc=f"{cur_filename} - Building edges")):

        if speed[graph.nodes[n]['y'], graph.nodes[n]['x']] == 0:
            continue

        neighbor = speed[
            max(0, graph.nodes[n]['y'] - 1):min(im_y, graph.nodes[n]['y'] + 2),
            max(0, graph.nodes[n]['x'] - 1):min(im_x, graph.nodes[n]['x'] + 2)
        ]

        if np.mean(neighbor) < 0.1:
            continue

        # ===================
        # GEO DISTANCE
        # ===================
        if params.edge_method == 'geo_dist':
            phi = np.ones_like(speed)
            phi[graph.nodes[n]['y'], graph.nodes[n]['x']] = -1
            tt = skfmm.travel_time(phi, speed, narrow=params.edge_dist_thresh)

            for n2 in node_list[i+1:]:
                geo_dist = tt[graph.nodes[n2]['y'], graph.nodes[n2]['x']]
                if geo_dist < params.edge_dist_thresh:
                    w = params.edge_dist_thresh / (params.edge_dist_thresh + geo_dist)
                    graph.add_edge(n, n2, weight=w)

        # ===================
        # EUCLIDEAN DISTANCE
        # ===================
        else:
            for n2 in node_list[i+1:]:
                eu_dist = (graph.nodes[n2]['y'] - graph.nodes[n]['y'])**2 + \
                          (graph.nodes[n2]['x'] - graph.nodes[n]['x'])**2

                if eu_dist < edge_dist_thresh_sq:
                    graph.add_edge(n, n2, weight=1.)

    # ==========================
    # Lưu output
    # ==========================
    util.visualize_graph(
        im, graph, show_graph=False, save_graph=True,
        num_nodes_each_type=[0, graph.number_of_nodes()],
        save_path=cur_vis_res_im_savepath)

    util.visualize_graph(
        gt_mask, graph, show_graph=False, save_graph=True,
        num_nodes_each_type=[0, graph.number_of_nodes()],
        save_path=cur_vis_res_mask_savepath)

    with open(cur_res_graph_savepath, 'wb') as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✔ Saved: {cur_res_graph_savepath}")

    graph.clear()


# ============================
#               MAIN
# ============================
if __name__ == "__main__":

    args = parse_args()
    print("\nARGS:", args)

    # =============================================
    # Tự suy ra im_root_path + cnn_result_root_path
    # =============================================
    input_path = args.input_img
    img_dir = os.path.dirname(input_path)

    if args.dataset == "STARE":
        im_root_path = "/content/data/STARE/images"
        cnn_result_root_path = "/content/VGN-update/GenGraph"

    elif args.dataset == "DRIVE":
        im_root_path = "../../DRIVE/all"
        cnn_result_root_path = "../new_exp/DRIVE_cnn/test"

    elif args.dataset == "CHASE_DB1":
        im_root_path = "../../CHASE_DB1/all"
        cnn_result_root_path = "../CHASE_cnn/test_resized_graph_gen"

    elif args.dataset == "HRF":
        im_root_path = "../../HRF/all_768"
        cnn_result_root_path = "../HRF_cnn/test"

    # =============================================
    # Tạo tuple input duy nhất
    # =============================================
    func_args = (input_path, im_root_path, cnn_result_root_path, args)

    generate_graph_using_srns(func_args)

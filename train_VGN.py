# updated by syshin (180825)
# do the following steps before running this script
# (1) run the script 'GenGraph/make_graph_db.py'
# to generate training/test graphs
# (2) place the generated graphs ('.graph_res')
# and cnn results ('_prob.png') in
# a new directory 'args.save_root/graph'
# --- BẮT ĐẦU SỬA LỖI ---
import tensorflow as tf
tf.compat.v1.disable_eager_execution()  # ← DÙNG CÁI NÀY
# --- KẾT THÚC SỬA LỖI ---
import numpy as np
import os
import pdb
import argparse
import skimage.io
import networkx as nx
import pickle as pkl
import multiprocessing
import sys
import skfmm
import tensorflow as tf
import json

import _init_paths
from config import cfg
from model import VesselSegmVGN as vessel_segm_vgn
import util
from train_CNN import load


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a vessel_segm_vgn network')
    parser.add_argument('--dataset', default='DRIVE', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1 or HRF', type=str)
    parser.add_argument('--use_multiprocessing', default='True', help='Whether to use the python multiprocessing module', type=str)
    parser.add_argument('--multiprocessing_num_proc', default=8, help='Number of CPU processes to use', type=int)
    parser.add_argument('--win_size', default=4, help='Window size for srns', type=int)  # for srns # [4,8,16]
    parser.add_argument('--edge_type', default='srns_geo_dist_binary',
                        help='Graph edge type: Can be srns_geo_dist_binary or srns_geo_dist_weighted', type=str)
    parser.add_argument('--edge_geo_dist_thresh', default=10, help='Threshold for geodesic distance', type=float)  # [10,20,40]
    parser.add_argument('--pretrained_model', default='../models/DRIVE/DRIU*/DRIU_DRIVE.ckpt',
                        help='Path for a pretrained model(.ckpt)', type=str)
    parser.add_argument('--save_root', default='../models/DRIVE/VGN_DRIVE',
                        help='Root path to save trained models and test results', type=str)

    ### cnn module related ###
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--cnn_loss_on', default='True', help='Whether to use a cnn loss for training', type=str)

    ### gnn module related ###
    parser.add_argument('--gnn_loss_on', default='True', help='Whether to use a gnn loss for training', type=str)
    parser.add_argument('--gnn_loss_weight', default=1., help='Relative weight on the gnn loss', type=float)
    parser.add_argument('--gnn_feat_dropout_prob', default=0.5, help='Dropout prob. for feat. in gnn layers', type=float)
    parser.add_argument('--gnn_att_dropout_prob', default=0.5, help='Dropout prob. for att. in gnn layers', type=float)
    # gat #
    parser.add_argument('--gat_n_heads', default='[4,4]', help='Numbers of heads in each layer', type=str)  # [4,1]
    parser.add_argument('--gat_hid_units', default='[16]', help='Numbers of hidden units per each attention head in each layer', type=str)
    parser.add_argument('--gat_use_residual', default='False', help='Whether to use residual learning in GAT', type=str)

    ### inference module related ###
    parser.add_argument('--norm_type', default=None, help='Norm. type', type=str)
    parser.add_argument('--use_enc_layer', default='False',
                        help='Whether to use additional conv. layers in the inference module', type=str)
    parser.add_argument('--infer_module_loss_masking_thresh', default=0.05,
                        help='Threshold for loss masking', type=float)
    parser.add_argument('--infer_module_kernel_size', default=3,
                        help='Conv. kernel size for the inference module', type=int)
    parser.add_argument('--infer_module_grad_weight', default=1.,
                        help='Relative weight of the grad. on the inference module', type=float)
    parser.add_argument('--infer_module_dropout_prob', default=0.1,
                        help='Dropout prob. for layers in the inference module', type=float)

    ### training (declared but not used) ###
    parser.add_argument('--do_simul_training', default='True',
                        help='Whether to train the gnn and inference modules simultaneously or not', type=str)
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int)
    parser.add_argument('--old_net_ft_lr', default=1e-02, help='Learning rate for fine-tuning of old parts of network', type=float)
    parser.add_argument('--new_net_lr', default=1e-02, help='Learning rate for a new part of network', type=float)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str)
    parser.add_argument('--lr_scheduling', default='pc', help='How to change the learning rate during training', type=str)  # [pc]
    parser.add_argument('--lr_decay_tp', default=1., help='When to decrease the lr during training', type=float)  # for pc
    parser.add_argument('--use_graph_update', default='True',
                        help='Whether to update graphs during training', type=str)
    parser.add_argument('--graph_update_period', default=10000, help='Graph update period', type=int)
    parser.add_argument('--use_fov_mask', default='True', help='Whether to use fov masks', type=str)

    args = parser.parse_args()

    # Convert string args to proper types
    def str_to_bool(s): return s.lower() in ('true', '1', 'yes')
    def str_to_list(s): return json.loads(s.replace("'", '"'))

    args.use_multiprocessing = str_to_bool(args.use_multiprocessing)
    args.cnn_loss_on = str_to_bool(args.cnn_loss_on)
    args.gnn_loss_on = str_to_bool(args.gnn_loss_on)
    args.gat_n_heads = str_to_list(args.gat_n_heads)
    args.gat_hid_units = str_to_list(args.gat_hid_units)
    args.gat_use_residual = str_to_bool(args.gat_use_residual)
    args.use_enc_layer = str_to_bool(args.use_enc_layer)
    args.do_simul_training = str_to_bool(args.do_simul_training)
    args.use_graph_update = str_to_bool(args.use_graph_update)
    args.use_fov_mask = str_to_bool(args.use_fov_mask)

    return args


def restore_from_pretrained_model(sess, saver, network, pretrained_model_path):
    splits = pretrained_model_path.split('/')
    if any('DRIU*' in s for s in splits):
        var_dict = {}
        for v in tf.compat.v1.trainable_variables():
            t_var_name = v.name
            highest_level_name = t_var_name[:t_var_name.find('/')]
            if highest_level_name in network.var_to_restore:
                if highest_level_name == 'img_output':
                    t_var_name_to_restore = 'output' + t_var_name[t_var_name.find('/'):t_var_name.rfind(':')]
                    var_dict[t_var_name_to_restore] = v
                elif ('gat' in highest_level_name) or ('post_cnn' in highest_level_name):
                    pass
                else:
                    var_dict[t_var_name[:t_var_name.rfind(':')]] = v

        loader = tf.train.Saver(var_list=var_dict)
        loader.restore(sess, pretrained_model_path)
    else:
        saver.restore(sess, pretrained_model_path)


def make_train_qual_res(args_tuple):
    img_name, fg_prob_map, temp_graph_save_path, args = args_tuple

    if 'srns' not in args.edge_type:
        raise NotImplementedError

    win_size_str = '%.2d_%.2d' % (args.win_size, args.edge_geo_dist_thresh)

    last_slash = img_name.rfind('/')
    cur_filename = img_name[last_slash + 1:]

    print('regenerating a graph for ' + cur_filename)

    temp = (fg_prob_map * 255).astype(int)
    cur_save_path = os.path.join(temp_graph_save_path, cur_filename + '_prob.png')
    skimage.io.imsave(cur_save_path, temp)

    cur_res_graph_savepath = os.path.join(temp_graph_save_path, cur_filename + '_' + win_size_str + '.graph_res')

    # find local maxima
    vesselness = fg_prob_map

    im_y = vesselness.shape[0]
    im_x = vesselness.shape[1]
    y_quan = list(range(0, im_y, args.win_size))
    y_quan = sorted(list(set(y_quan) | {im_y}))
    x_quan = list(range(0, im_x, args.win_size))
    x_quan = sorted(list(set(x_quan) | {im_x}))

    max_val = []
    max_pos = []
    for y_idx in range(len(y_quan) - 1):
        for x_idx in range(len(x_quan) - 1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx + 1], x_quan[x_idx]:x_quan[x_idx + 1]]
            if np.sum(cur_patch) == 0:
                max_val.append(0)
                max_pos.append((y_quan[y_idx] + cur_patch.shape[0] // 2, x_quan[x_idx] + cur_patch.shape[1] // 2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx] + temp[0], x_quan[x_idx] + temp[1]))

    graph = nx.Graph()

    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=node_y, x=node_x, label=node_idx)
        print('node label', node_idx, 'pos', (node_y, node_x), 'added')

    speed = vesselness

    node_list = list(graph.nodes)
    for i, n in enumerate(node_list):
        phi = np.ones_like(speed)
        phi[graph.nodes[n]['y'], graph.nodes[n]['x']] = -1
        if speed[graph.nodes[n]['y'], graph.nodes[n]['x']] == 0:
            continue

        neighbor = speed[max(0, graph.nodes[n]['y'] - 1):min(im_y, graph.nodes[n]['y'] + 2),
                         max(0, graph.nodes[n]['x'] - 1):min(im_x, graph.nodes[n]['x'] + 2)]
        if np.mean(neighbor) < 0.1:
            continue

        tt = skfmm.travel_time(phi, speed, narrow=args.edge_geo_dist_thresh)  # travel time

        for n_comp in node_list[i + 1:]:
            geo_dist = tt[graph.nodes[n_comp]['y'], graph.nodes[n_comp]['x']]  # travel time
            if geo_dist < args.edge_geo_dist_thresh:
                weight = args.edge_geo_dist_thresh / (args.edge_geo_dist_thresh + geo_dist)
                graph.add_edge(n, n_comp, weight=weight)
                print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')

    # save as files
    nx.write_gpickle(graph, cur_res_graph_savepath, protocol=pkl.HIGHEST_PROTOCOL)

    graph.clear()


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == 'DRIVE':
        im_root_path = '../DRIVE/all'
        train_set_txt_path = cfg.TRAIN.DRIVE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
    elif args.dataset == 'STARE':
        im_root_path = '../content/data/STARE/images'
        train_set_txt_path = "/content/data/STARE/list/train.txt"
        test_set_txt_path = "/content/data/STARE/list/test.txt"
    elif args.dataset == 'CHASE_DB1':
        im_root_path = '../CHASE_DB1/all'
        train_set_txt_path = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    elif args.dataset == 'HRF':
        im_root_path = '../HRF/all_768'
        train_set_txt_path = cfg.TRAIN.HRF_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.HRF_SET_TXT_PATH

    if args.use_multiprocessing:
        pool = multiprocessing.Pool(processes=args.multiprocessing_num_proc)

    model_save_path = os.path.join(args.save_root, cfg.TRAIN.MODEL_SAVE_PATH) if len(args.save_root) > 0 else cfg.TRAIN.MODEL_SAVE_PATH
    res_save_path = os.path.join(args.save_root, cfg.TEST.RES_SAVE_PATH) if len(args.save_root) > 0 else cfg.TEST.RES_SAVE_PATH
    temp_graph_save_path = os.path.join(args.save_root, cfg.TRAIN.TEMP_GRAPH_SAVE_PATH) if len(args.save_root) > 0 else cfg.TRAIN.TEMP_GRAPH_SAVE_PATH

    for path in [args.save_root, model_save_path, res_save_path]:
        if len(args.save_root) > 0 and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    with open(train_set_txt_path) as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]

    if args.dataset == 'HRF':
        test_img_names = [test_img_names[x] for x in range(7, len(test_img_names), 20)]

    len_train = len(train_img_names)
    len_test = len(test_img_names)

    # revise "train_img_names" and "test_img_names"
    for i in range(len_train):
        temp = train_img_names[i]
        last_slash = temp.rfind('/')
        train_img_names[i] = temp_graph_save_path + temp[last_slash:]

    for i in range(len_test):
        temp = test_img_names[i]
        last_slash = temp.rfind('/')
        test_img_names[i] = temp_graph_save_path + temp[last_slash:]

    data_layer_train = util.GraphDataLayer(train_img_names, is_training=True,
                                           edge_type=args.edge_type,
                                           win_size=args.win_size, edge_geo_dist_thresh=args.edge_geo_dist_thresh)
    data_layer_test = util.GraphDataLayer(test_img_names, is_training=False,
                                          edge_type=args.edge_type,
                                          win_size=args.win_size, edge_geo_dist_thresh=args.edge_geo_dist_thresh)

    network = vessel_segm_vgn(args, None)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    saver = tf.compat.v1.train.Saver(max_to_keep=100)
    summary_writer = tf.compat.v1.summary.FileWriter(model_save_path, sess.graph)

    sess.run(tf.compat.v1.global_variables_initializer())
    network = vessel_segm_vgn(args, None)
    if args.pretrained_model is not None:
        print("Loading model...")
        ext_str = args.pretrained_model[args.pretrained_model.rfind('.') + 1:]
        if ext_str == 'ckpt':
            restore_from_pretrained_model(sess, saver, network, args.pretrained_model)
        elif ext_str == 'npy':
            load(args.pretrained_model, sess, ignore_missing=True)

    f_log = open(os.path.join(model_save_path, 'log.txt'), 'w')
    f_log.write(str(args) + '\n')
    f_log.flush()
    last_snapshot_iter = -1
    timer = util.Timer()

    # for graph update
    required_num_iters_for_train_set_update = int(np.ceil(float(len_train) / cfg.TRAIN.GRAPH_BATCH_SIZE))
    required_num_iters_for_test_set_update = int(np.ceil(float(len_test) / cfg.TRAIN.GRAPH_BATCH_SIZE))
    if args.use_graph_update:
        next_update_start = args.graph_update_period
        next_update_end = next_update_start + required_num_iters_for_train_set_update - 1
    else:
        next_update_start = sys.maxsize
        next_update_end = sys.maxsize

    train_loss_list = []
    train_cnn_loss_list = []
    train_gnn_loss_list = []
    train_infer_module_loss_list = []
    test_loss_list = []
    test_cnn_loss_list = []
    test_gnn_loss_list = []
    test_infer_module_loss_list = []
    graph_update_func_arg = []
    test_loss_logs = []
    print("Training the model...")
    for iter in range(args.max_iters):

        timer.tic()

        # get one batch
        img_list, blobs_train = data_layer_train.forward()

        img = blobs_train['img']
        label = blobs_train['label']
        if args.use_fov_mask:
            fov_mask = blobs_train['fov']
        else:
            fov_mask = np.ones(label.shape, dtype=label.dtype)

        graph = blobs_train['graph']
        num_of_nodes_list = blobs_train['num_of_nodes_list']

        node_byxs = util.get_node_byx_from_graph(graph, num_of_nodes_list)
        probmap = blobs_train['probmap']
        pixel_weights = fov_mask * ((probmap >= args.infer_module_loss_masking_thresh) | label)
        pixel_weights = pixel_weights.astype(float)

        if 'geo_dist_weighted' in args.edge_type:
            adj = nx.adjacency_matrix(graph)
        else:
            adj = nx.adjacency_matrix(graph, weight=None)

        adj_norm = util.preprocess_graph_gat(adj.toarray().astype(float))

        is_lr_flipped = False
        is_ud_flipped = False
        rot90_num = 0

        if blobs_train['vec_aug_on'][0]:
            is_lr_flipped = True
        if blobs_train['vec_aug_on'][1]:
            is_ud_flipped = True
        if blobs_train['vec_aug_on'][2]:
            rot90_num = blobs_train['rot_angle'] / 90

        if args.lr_scheduling == 'pc':
            cur_lr = sess.run([network.lr_handler])[0]
        else:
            raise NotImplementedError

        _, loss_val, \
        cnn_fg_prob_mat, \
        cnn_loss_val, cnn_accuracy_val, cnn_precision_val, cnn_recall_val, \
        gnn_loss_val, gnn_accuracy_val, \
        infer_module_fg_prob_mat, \
        infer_module_loss_val, infer_module_accuracy_val, infer_module_precision_val, infer_module_recall_val, \
        node_logits, node_labels = sess.run(
            [network.train_op, network.loss,
             network.img_fg_prob,
             network.cnn_loss, network.cnn_accuracy, network.cnn_precision, network.cnn_recall,
             network.gnn_loss, network.gnn_accuracy,
             network.post_cnn_img_fg_prob,
             network.post_cnn_loss, network.post_cnn_accuracy, network.post_cnn_precision, network.post_cnn_recall,
             network.node_logits, network.node_labels
             ],
            feed_dict={
                network.imgs: img,
                network.labels: label,
                network.fov_masks: fov_mask,
                network.node_byxs: node_byxs,
                network.adj: adj_norm,
                network.pixel_weights: pixel_weights,
                network.gnn_feat_dropout: args.gnn_feat_dropout_prob,
                network.gnn_att_dropout: args.gnn_att_dropout_prob,
                network.post_cnn_dropout: args.infer_module_dropout_prob,
                network.is_lr_flipped: is_lr_flipped,
                network.is_ud_flipped: is_ud_flipped,
                network.rot90_num: rot90_num,
                network.learning_rate: cur_lr
            })

        timer.toc()
        train_loss_list.append(loss_val)
        train_cnn_loss_list.append(cnn_loss_val)
        train_gnn_loss_list.append(gnn_loss_val)
        train_infer_module_loss_list.append(infer_module_loss_val)

        if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
            print('iter: %d / %d, loss: %.4f' % (iter + 1, args.max_iters, loss_val))
            print('cnn_loss: %.4f, cnn_accuracy: %.4f, cnn_precision: %.4f, cnn_recall: %.4f' %
                  (cnn_loss_val, cnn_accuracy_val, cnn_precision_val, cnn_recall_val))
            print('gnn_loss: %.4f, gnn_accuracy: %.4f' % (gnn_loss_val, gnn_accuracy_val))
            print('infer_module_loss: %.4f, infer_module_accuracy: %.4f, infer_module_precision: %.4f, infer_module_recall: %.4f' %
                  (infer_module_loss_val, infer_module_accuracy_val, infer_module_precision_val, infer_module_recall_val))
            print('speed: {:.3f}s / iter'.format(timer.average_time))

        if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
            last_snapshot_iter = iter
            filename = os.path.join(model_save_path, ('iter_{:d}'.format(iter + 1) + '.ckpt'))
            saver.save(sess, filename)
            print('Wrote snapshot to: {:s}'.format(filename))

        if (iter + 1) == next_update_start - 1:
            data_layer_train.reinit(train_img_names, is_training=False,
                                    edge_type=args.edge_type,
                                    win_size=args.win_size, edge_geo_dist_thresh=args.edge_geo_dist_thresh)

        if ((iter + 1) < args.max_iters) and ((iter + 1) >= next_update_start) and ((iter + 1) <= next_update_end):
            cur_batch_size = len(img_list)
            reshaped_fg_prob_map = infer_module_fg_prob_mat.reshape((cur_batch_size, infer_module_fg_prob_mat.shape[1], infer_module_fg_prob_mat.shape[2]))

            for j in range(cur_batch_size):
                graph_update_func_arg.append((img_list[j], reshaped_fg_prob_map[j, :, :], temp_graph_save_path, args))

        if (iter + 1) == next_update_end:
            if args.use_multiprocessing:
                pool.map(make_train_qual_res, graph_update_func_arg)
            else:
                for x in graph_update_func_arg:
                    make_train_qual_res(x)
            graph_update_func_arg = []

            data_layer_train.reinit(train_img_names, is_training=True,
                                    edge_type=args.edge_type,
                                    win_size=args.win_size, edge_geo_dist_thresh=args.edge_geo_dist_thresh)
            next_update_start = next_update_start + args.graph_update_period
            next_update_end = next_update_start + required_num_iters_for_train_set_update - 1

        if (iter + 1) % cfg.TRAIN.TEST_ITERS == 0:
            all_cnn_labels = np.zeros((0,))
            all_cnn_preds = np.zeros((0,))
            all_gnn_labels = np.zeros((0,))
            all_gnn_preds = np.zeros((0,))
            all_infer_module_preds = np.zeros((0,))

            for _ in range(int(np.ceil(float(len_test) / cfg.TRAIN.GRAPH_BATCH_SIZE))):
                img_list, blobs_test = data_layer_test.forward()

                img = blobs_test['img']
                label = blobs_test['label']
                if args.use_fov_mask:
                    fov_mask = blobs_test['fov']
                else:
                    fov_mask = np.ones(label.shape, dtype=label.dtype)

                graph = blobs_test['graph']
                num_of_nodes_list = blobs_test['num_of_nodes_list']

                node_byxs = util.get_node_byx_from_graph(graph, num_of_nodes_list)
                pixel_weights = fov_mask

                if 'geo_dist_weighted' in args.edge_type:
                    adj = nx.adjacency_matrix(graph)
                else:
                    adj = nx.adjacency_matrix(graph, weight=None)

                adj_norm = util.preprocess_graph_gat(adj.toarray().astype(float))

                loss_val, \
                cnn_fg_prob_mat, cnn_loss_val, \
                gnn_labels, gnn_prob_vec, gnn_loss_val, \
                infer_module_fg_prob_mat, infer_module_loss_val = sess.run(
                    [network.loss,
                     network.img_fg_prob, network.cnn_loss,
                     network.node_labels, network.gnn_prob, network.gnn_loss,
                     network.post_cnn_img_fg_prob, network.post_cnn_loss],
                    feed_dict={
                        network.imgs: img,
                        network.labels: label,
                        network.fov_masks: fov_mask,
                        network.node_byxs: node_byxs,
                        network.adj: adj_norm,
                        network.pixel_weights: pixel_weights,
                        network.is_lr_flipped: False,
                        network.is_ud_flipped: False
                    })

                cnn_fg_prob_mat = cnn_fg_prob_mat * fov_mask.astype(float)
                infer_module_fg_prob_mat = infer_module_fg_prob_mat * fov_mask.astype(float)

                test_loss_list.append(loss_val)
                test_cnn_loss_list.append(cnn_loss_val)
                test_gnn_loss_list.append(gnn_loss_val)
                test_infer_module_loss_list.append(infer_module_loss_val)

                all_cnn_labels = np.concatenate((all_cnn_labels, np.reshape(label, (-1))))
                all_cnn_preds = np.concatenate((all_cnn_preds, np.reshape(cnn_fg_prob_mat, (-1))))

                all_gnn_labels = np.concatenate((all_gnn_labels, gnn_labels))
                all_gnn_preds = np.concatenate((all_gnn_preds, gnn_prob_vec))

                all_infer_module_preds = np.concatenate((all_infer_module_preds, np.reshape(infer_module_fg_prob_mat, (-1))))

                cur_batch_size = len(img_list)
                reshaped_cnn_fg_prob_map = cnn_fg_prob_mat.reshape((cur_batch_size, cnn_fg_prob_mat.shape[1], cnn_fg_prob_mat.shape[2]))
                reshaped_infer_module_fg_prob_mat = infer_module_fg_prob_mat.reshape((cur_batch_size, infer_module_fg_prob_mat.shape[1], infer_module_fg_prob_mat.shape[2]))
                for j in range(cur_batch_size):
                    cur_img_name = img_list[j]
                    last_slash = cur_img_name.rfind('/')
                    cur_img_name = cur_img_name[last_slash:]

                    cur_cnn_fg_prob_map = reshaped_cnn_fg_prob_map[j, :, :]
                    cur_infer_module_fg_prob_map = reshaped_infer_module_fg_prob_mat[j, :, :]

                    cur_map = (cur_cnn_fg_prob_map * 255).astype(int)
                    cur_save_path = res_save_path + cur_img_name + '_prob_cnn.png'
                    skimage.io.imsave(cur_save_path, cur_map)

                    cur_map = (cur_infer_module_fg_prob_map * 255).astype(int)
                    cur_save_path = res_save_path + cur_img_name + '_prob_infer_module.png'
                    skimage.io.imsave(cur_save_path, cur_map)

            cnn_auc_test, cnn_ap_test = util.get_auc_ap_score(all_cnn_labels, all_cnn_preds)
            all_cnn_labels_bin = all_cnn_labels.astype(bool).copy()
            all_cnn_preds_bin = all_cnn_preds >= 0.5
            all_cnn_correct = all_cnn_labels_bin == all_cnn_preds_bin
            cnn_acc_test = np.mean(all_cnn_correct.astype(np.float32))

            gnn_auc_test, gnn_ap_test = util.get_auc_ap_score(all_gnn_labels, all_gnn_preds)
            all_gnn_labels_bin = all_gnn_labels.astype(bool).copy()
            all_gnn_preds_bin = all_gnn_preds >= 0.5
            all_gnn_correct = all_gnn_labels_bin == all_gnn_preds_bin
            gnn_acc_test = np.mean(all_gnn_correct.astype(np.float32))

            infer_module_auc_test, infer_module_ap_test = util.get_auc_ap_score(all_cnn_labels, all_infer_module_preds)
            all_infer_module_preds_bin = all_infer_module_preds >= 0.5
            all_infer_module_correct = all_cnn_labels_bin == all_infer_module_preds_bin
            infer_module_acc_test = np.mean(all_infer_module_correct.astype(np.float32))

            summary = tf.Summary()
            summary.value.add(tag="train_loss", simple_value=float(np.mean(train_loss_list)))
            summary.value.add(tag="train_cnn_loss", simple_value=float(np.mean(train_cnn_loss_list)))
            summary.value.add(tag="train_gnn_loss", simple_value=float(np.mean(train_gnn_loss_list)))
            summary.value.add(tag="train_infer_module_loss", simple_value=float(np.mean(train_infer_module_loss_list)))
            summary.value.add(tag="test_loss", simple_value=float(np.mean(test_loss_list)))
            summary.value.add(tag="test_cnn_loss", simple_value=float(np.mean(test_cnn_loss_list)))
            summary.value.add(tag="test_gnn_loss", simple_value=float(np.mean(test_gnn_loss_list)))
            summary.value.add(tag="test_infer_module_loss", simple_value=float(np.mean(test_infer_module_loss_list)))
            summary.value.add(tag="test_cnn_acc", simple_value=float(cnn_acc_test))
            summary.value.add(tag="test_cnn_auc", simple_value=float(cnn_auc_test))
            summary.value.add(tag="test_cnn_ap", simple_value=float(cnn_ap_test))
            summary.value.add(tag="test_gnn_acc", simple_value=float(gnn_acc_test))
            summary.value.add(tag="test_gnn_auc", simple_value=float(gnn_auc_test))
            summary.value.add(tag="test_gnn_ap", simple_value=float(gnn_ap_test))
            summary.value.add(tag="test_infer_module_acc", simple_value=float(infer_module_acc_test))
            summary.value.add(tag="test_infer_module_auc", simple_value=float(infer_module_auc_test))
            summary.value.add(tag="test_infer_module_ap", simple_value=float(infer_module_ap_test))
            summary.value.add(tag="lr", simple_value=float(cur_lr))
            summary_writer.add_summary(summary, global_step=iter + 1)
            summary_writer.flush()

            print('iter: %d / %d, train_loss: %.4f, train_cnn_loss: %.4f, train_gnn_loss: %.4f, train_infer_module_loss: %.4f' %
                  (iter + 1, args.max_iters, np.mean(train_loss_list), np.mean(train_cnn_loss_list), np.mean(train_gnn_loss_list), np.mean(train_infer_module_loss_list)))
            print('iter: %d / %d, test_loss: %.4f, test_cnn_loss: %.4f, test_gnn_loss: %.4f, test_infer_module_loss: %.4f' %
                  (iter + 1, args.max_iters, np.mean(test_loss_list), np.mean(test_cnn_loss_list), np.mean(test_gnn_loss_list), np.mean(test_infer_module_loss_list)))
            print('test_cnn_acc: %.4f, test_cnn_auc: %.4f, test_cnn_ap: %.4f' % (cnn_acc_test, cnn_auc_test, cnn_ap_test))
            print('test_gnn_acc: %.4f, test_gnn_auc: %.4f, test_gnn_ap: %.4f' % (gnn_acc_test, gnn_auc_test, gnn_ap_test))
            print('test_infer_module_acc: %.4f, test_infer_module_auc: %.4f, test_infer_module_ap: %.4f' % (infer_module_acc_test, infer_module_auc_test, infer_module_ap_test))
            print('lr: %.8f' % (cur_lr))

            f_log.write('iter: ' + str(iter + 1) + ' / ' + str(args.max_iters) + '\n')
            f_log.write('train_loss ' + str(np.mean(train_loss_list)) + '\n')
            f_log.write('train_cnn_loss ' + str(np.mean(train_cnn_loss_list)) + '\n')
            f_log.write('train_gnn_loss ' + str(np.mean(train_gnn_loss_list)) + '\n')
            f_log.write('train_infer_module_loss ' + str(np.mean(train_infer_module_loss_list)) + '\n')
            f_log.write('iter: ' + str(iter + 1) + ' / ' + str(args.max_iters) + '\n')
            f_log.write('test_loss ' + str(np.mean(test_loss_list)) + '\n')
            f_log.write('test_cnn_loss ' + str(np.mean(test_cnn_loss_list)) + '\n')
            f_log.write('test_gnn_loss ' + str(np.mean(test_gnn_loss_list)) + '\n')
            f_log.write('test_infer_module_loss ' + str(np.mean(test_infer_module_loss_list)) + '\n')
            f_log.write('test_cnn_acc ' + str(cnn_acc_test) + '\n')
            f_log.write('test_cnn_auc ' + str(cnn_auc_test) + '\n')
            f_log.write('test_cnn_ap ' + str(cnn_ap_test) + '\n')
            f_log.write('test_gnn_acc ' + str(gnn_acc_test) + '\n')
            f_log.write('test_gnn_auc ' + str(gnn_auc_test) + '\n')
            f_log.write('test_gnn_ap ' + str(gnn_ap_test) + '\n')
            f_log.write('test_infer_module_acc ' + str(infer_module_acc_test) + '\n')
            f_log.write('test_infer_module_auc ' + str(infer_module_auc_test) + '\n')
            f_log.write('test_infer_module_ap ' + str(infer_module_ap_test) + '\n')
            f_log.write('lr ' + str(cur_lr) + '\n')
            f_log.flush()

            test_loss_logs.append(float(np.mean(test_loss_list)))

            train_loss_list = []
            train_cnn_loss_list = []
            train_gnn_loss_list = []
            train_infer_module_loss_list = []
            test_loss_list = []
            test_cnn_loss_list = []
            test_gnn_loss_list = []
            test_infer_module_loss_list = []

            all_cnn_labels = np.zeros((0,))
            all_cnn_preds = np.zeros((0,))
            all_gnn_labels = np.zeros((0,))
            all_gnn_preds = np.zeros((0,))
            all_infer_module_preds = np.zeros((0,))

    if last_snapshot_iter != iter:
        filename = os.path.join(model_save_path, ('iter_{:d}'.format(iter + 1) + '.ckpt'))
        saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    f_log.close()
    sess.close()
    if args.use_multiprocessing:
        pool.terminate()
    print("Training complete.")
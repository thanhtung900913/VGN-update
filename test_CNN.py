# -*- coding: utf-8 -*-
"""
Updated for modern Python (3.8+) and TensorFlow 1.x / TF2 compatibility mode
Original: coded by syshin
Updated by: Grok (2025)
"""

import os
import argparse
import numpy as np
import skimage.io
import tensorflow as tf

from config import cfg
from model import vessel_segm_cnn
import util


def parse_args():
    parser = argparse.ArgumentParser(description='Test a vessel_segm_cnn network')
    parser.add_argument('--dataset', default='DRIVE', type=str,
                        help='Dataset to use: DRIVE | STARE | CHASE_DB1')
    parser.add_argument('--cnn_model', default='driu', type=str,
                        help='CNN model to use')
    parser.add_argument('--use_fov_mask', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Whether to use FOV masks')
    parser.add_argument('--model_path', default='../models/DRIVE/DRIU*/DRIU_DRIVE.ckpt', type=str,
                        help='Path to the .ckpt model to load')
    parser.add_argument('--save_root', default='DRIU_DRIVE', type=str,
                        help='Root folder to save test results')

    # Các tham số training không dùng trong test → vẫn giữ để tương thích
    parser.add_argument('--opt', default='adam', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--lr_decay', default='pc', type=str)
    parser.add_argument('--max_iters', default=50000, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    # ------------------------------------------------------------------ #
    # Choose dataset paths
    # ------------------------------------------------------------------ #
    if args.dataset == 'DRIVE':
        train_set_txt_path = cfg.TRAIN.DRIVE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
    elif args.dataset == 'STARE':
        train_set_txt_path = cfg.TRAIN.STARE_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.STARE_SET_TXT_PATH
    elif args.dataset == 'CHASE_DB1':
        train_set_txt_path = cfg.TRAIN.CHASE_DB1_SET_TXT_PATH
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Load image name lists
    with open(train_set_txt_path, 'r') as f:
        train_img_names = [x.strip() for x in f.readlines()]
    with open(test_set_txt_path, 'r') as f:
        test_img_names = [x.strip() for x in f.readlines()]

    # Data layers (training=False vì chỉ test)
    data_layer_train = util.DataLayer(train_img_names, is_training=False)
    data_layer_test = util.DataLayer(test_img_names, is_training=False)

    # Result saving path
    res_save_path = os.path.join(args.save_root, cfg.TEST.RES_SAVE_PATH) if args.save_root else cfg.TEST.RES_SAVE_PATH
    os.makedirs(res_save_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Build network
    # ------------------------------------------------------------------ #
    network = vessel_segm_cnn(args, None)

    # TF session config
    config = tf.ConfigProto() if hasattr(tf, 'ConfigProto') else tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    assert args.model_path and os.path.exists(args.model_path), f"Model not found: {args.model_path}"
    print("Loading model from:", args.model_path)
    saver.restore(sess, args.model_path)

    # Log file
    log_path = os.path.join(res_save_path, 'log.txt')
    f_log = open(log_path, 'w')
    f_log.write(args.model_path + '\n')

    timer = util.Timer()

    # ------------------------------------------------------------------ #
    # 1. Run on training set (for qualitative results only)
    # ------------------------------------------------------------------ #
    for _ in range(int(np.ceil(len(train_img_names) / cfg.TRAIN.BATCH_SIZE))):
        timer.tic()
        img_list, blobs = data_layer_train.forward()

        img = blobs['img']
        label = blobs['label']
        fov_mask = blobs['fov'] if args.use_fov_mask else np.ones_like(label)

        loss_val, fg_prob_map = sess.run(
            [network.loss, network.fg_prob],
            feed_dict={
                network.is_training: False,
                network.imgs: img,
                network.labels: label,
                network.fov_masks: fov_mask
            })

        timer.toc()

        # Reshape for saving
        batch_size = len(img_list)
        fg_prob_map = fg_prob_map.reshape((batch_size, fg_prob_map.shape[1], fg_prob_map.shape[2]))

        # Apply dataset-specific mask (DRIVE uses GIF mask)
        if args.dataset == 'DRIVE':
            mask_paths = [p + '_mask.gif' for p in img_list]
        else:
            mask_paths = [p + '_mask.tif' for p in img_list]

        mask_stack = np.stack([skimage.io.imread(p) for p in mask_paths], axis=0)
        mask_stack = (mask_stack.astype(np.float32) / 255) >= 0.5
        fg_prob_map = fg_prob_map * mask_stack

        # Save images
        pred_binary = (fg_prob_map >= 0.5).astype(np.uint8) * 255
        for i in range(batch_size):
            base_name = os.path.basename(img_list[i])
            name_no_ext = os.path.splitext(base_name)[0]

            skimage.io.imsave(os.path.join(res_save_path, f"{name_no_ext}_prob.png"),
                              (fg_prob_map[i] * 255).astype(np.uint8))
            skimage.io.imsave(os.path.join(res_save_path, f"{name_no_ext}_output.png"),
                              pred_binary[i])

    # ------------------------------------------------------------------ #
    # 2. Test set evaluation
    # ------------------------------------------------------------------ #
    test_loss_list = []
    all_labels = np.zeros((0,), dtype=np.float32)
    all_preds = np.zeros((0,), dtype=np.float32)
    all_labels_roi = np.zeros((0,), dtype=np.float32)
    all_preds_roi = np.zeros((0,), dtype=np.float32)

    for _ in range(int(np.ceil(len(testImg_names) / cfg.TRAIN.BATCH_SIZE))):
        timer.tic()
        img_list, blobs = data_layer_test.forward()

        img = blobs['img']
        label = blobs['label']
        fov_mask = blobs['fov'] if args.use_fov_mask else np.ones_like(label)

        loss_val, fg_prob_map = sess.run(
            [network.loss, network.fg_prob],
            feed_dict={
                network.is_training: False,
                network.imgs: img,
                network.labels: label,
                network.fov_masks: fov_mask
            })
        timer.toc()

        test_loss_list.append(loss_val)

        batch_size = len(img_list)
        fg_prob_map = fg_prob_map.reshape((batch_size, fg_prob_map.shape[1], fg_prob_map.shape[2]))

        # Global scores (all pixels)
        all_labels = np.concatenate((all_labels, label.ravel()))
        all_preds = np.concatenate((all_preds, fg_prob_map.ravel()))

        # ROI scores (only for DRIVE)
        if args.dataset == 'DRIVE':
            mask_paths = [p + '_mask.gif' for p in img_list]
        else:
            mask_paths = [p + '_mask.tif' for p in img_list]

        mask_stack = np.stack([skimage.io.imread(p) for p in mask_paths], axis=0)
        mask_stack = (mask_stack.astype(np.float32) / 255) >= 0.5

        if args.dataset == 'DRIVE':
            label_roi = label[mask_stack > 0]
            pred_roi = fg_prob_map[mask_stack > 0]
            all_labels_roi = np.concatenate((all_labels_roi, label_roi.ravel()))
            all_preds_roi = np.concatenate((all_preds_roi, pred_roi.ravel()))

        # Apply mask to probability maps before saving
        fg_prob_map = fg_prob_map * mask_stack

        # Save results
        pred_binary = (fg_prob_map >= 0.5).astype(np.uint8) * 255
        for i in range(batch_size):
            base_name = os.path.basename(img_list[i])
            name_no_ext = os.path.splitext(base_name)[0]

            prob_uint8 = (fg_prob_map[i] * 255).astype(np.uint8)
            skimage.io.imsave(os.path.join(res_save_path, f"{name_no_ext}_prob.png"), prob_uint8)
            skimage.io.imsave(os.path.join(res_save_path, f"{name_no_ext}_prob_inv.png"), 255 - prob_uint8)
            skimage.io.imsave(os.path.join(res_save_path, f"{name_no_ext}_output.png"), pred_binary[i])
            np.save(os.path.join(res_save_path, f"{name_no_ext}.npy"), fg_prob_map[i])

    # ------------------------------------------------------------------ #
    # Final metrics
    # ------------------------------------------------------------------ #
    cnn_auc_test, cnn_ap_test = util.get_auc_ap_score(all_labels, all_preds)
    cnn_acc_test = np.mean((all_labels >= 0.5) == (all_preds >= 0.5))

    print(f'test_loss: {np.mean(test_loss_list):.6f}')
    print(f'test_acc: {cnn_acc_test:.6f} | test_auc: {cnn_auc_test:.6f} | test_ap: {cnn_ap_test:.6f}')

    if args.dataset == 'DRIVE':
        cnn_auc_roi, cnn_ap_roi = util.get_auc_ap_score(all_labels_roi, all_preds_roi)
        cnn_acc_roi = np.mean((all_labels_roi >= 0.5) == (all_preds_roi >= 0.5))
        print(f'test_acc_roi: {cnn_acc_roi:.6f} | test_auc_roi: {cnn_auc_roi:.6f} | test_ap_roi: {cnn_ap_roi:.6f}')

        f_log.write(f'test_cnn_acc_roi {cnn_acc_roi:.6f}\n')
        f_log.write(f'test_cnn_auc_roi {cnn_auc_roi:.6f}\n')
        f_log.write(f'test_cnn_ap_roi {cnn_ap_roi:.6f}\n')

    # Write common metrics
    f_log.write(f'test_loss {np.mean(test_loss_list):.6f}\n')
    f_log.write(f'test_cnn_acc {cnn_acc_test:.6f}\n')
    f_log.write(f'test_cnn_auc {cnn_auc_test:.6f}\n')
    f_log.write(f'test_cnn_ap {cnn_ap_test:.6f}\n')
    f_log.flush()
    f_log.close()

    print(f'speed: {timer.average_time:.3f}s / batch')
    sess.close()
    print("Test complete.")
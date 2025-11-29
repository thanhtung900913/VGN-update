# updated by syshin (180829)

import numpy as np
import os
import pdb
import argparse
import skimage.io
import networkx as nx
import pickle as pkl
import multiprocessing
import skfmm
import skimage.transform
import tensorflow as tf
from tqdm import tqdm
tf.compat.v1.disable_eager_execution()
from config import cfg
from model import vessel_segm_vgn
import util


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a vessel_segm_vgn network')
    parser.add_argument('--dataset', default='CHASE_DB1', help='Dataset to use: Can be DRIVE or STARE or CHASE_DB1', type=str)
    #parser.add_argument('--use_multiprocessing', action='store_true', default=False, help='Whether to use the python multiprocessing module')
    parser.add_argument('--use_multiprocessing', default=True, help='Whether to use the python multiprocessing module', type=bool)
    parser.add_argument('--multiprocessing_num_proc', default=8, help='Number of CPU processes to use', type=int)
    parser.add_argument('--win_size', default=16, help='Window size for srns', type=int) # for srns # [4,8,16]
    parser.add_argument('--edge_type', default='srns_geo_dist_binary', \
                        help='Graph edge type: Can be srns_geo_dist_binary or srns_geo_dist_weighted', type=str)
    parser.add_argument('--edge_geo_dist_thresh', default=40, help='Threshold for geodesic distance', type=float) # [10,20,40]
    parser.add_argument('--model_path', default='../models/CHASE_DB1/VGN/win_size=16/VGN_CHASE.ckpt', \
                        help='Path for a trained model(.ckpt)', type=str)
    parser.add_argument('--save_root', default='../models/CHASE_DB1/VGN/win_size=16', \
                        help='Root path to save test results', type=str)
    
    ### cnn module related ###    
    parser.add_argument('--cnn_model', default='driu', help='CNN model to use', type=str)
    parser.add_argument('--cnn_loss_on', default=True, help='Whether to use a cnn loss for training', type=bool)
    
    ### gnn module related ###
    parser.add_argument('--gnn_loss_on', default=True, help='Whether to use a gnn loss for training', type=bool)
    parser.add_argument('--gnn_loss_weight', default=1., help='Relative weight on the gnn loss', type=float)
    # gat #
    parser.add_argument('--gat_n_heads', default=[4,4], help='Numbers of heads in each layer', type=list) # [4,1]
    #parser.add_argument('--gat_n_heads', nargs='+', help='Numbers of heads in each layer', type=int) # [4,1]
    parser.add_argument('--gat_hid_units', default=[16], help='Numbers of hidden units per each attention head in each layer', type=list)
    #parser.add_argument('--gat_hid_units', nargs='+', help='Numbers of hidden units per each attention head in each layer', type=int)
    parser.add_argument('--gat_use_residual', action='store_true', default=False, help='Whether to use residual learning in GAT')    
    
    ### inference module related ###
    parser.add_argument('--norm_type', default=None, help='Norm. type', type=str)
    parser.add_argument('--use_enc_layer', action='store_true', default=False, \
                        help='Whether to use additional conv. layers in the inference module')
    parser.add_argument('--infer_module_loss_masking_thresh', default=0.05, \
                        help='Threshold for loss masking', type=float)
    parser.add_argument('--infer_module_kernel_size', default=3, \
                        help='Conv. kernel size for the inference module', type=int)
    parser.add_argument('--infer_module_grad_weight', default=1., \
                        help='Relative weight of the grad. on the inference module', type=float)

    ### training (declared but not used) ###
    parser.add_argument('--do_simul_training', default=True, \
                        help='Whether to train the gnn and inference modules simultaneously or not', type=bool)
    parser.add_argument('--max_iters', default=50000, help='Maximum number of iterations', type=int)
    parser.add_argument('--old_net_ft_lr', default=0., help='Learnining rate for fine-tuning of old parts of network', type=float)
    parser.add_argument('--new_net_lr', default=1e-02, help='Learnining rate for a new part of network', type=float)
    parser.add_argument('--opt', default='adam', help='Optimizer to use: Can be sgd or adam', type=str)
    parser.add_argument('--lr_scheduling', default='pc', help='How to change the learning rate during training', type=str)
    parser.add_argument('--lr_decay_tp', default=1., help='When to decrease the lr during training', type=float) # for pc


    args = parser.parse_args()
    return args
    
    
def make_graph_using_srns(args):
    fg_prob_map, edge_type, win_size, edge_geo_dist_thresh, img_path = args
    # Xác định tên file graph sẽ lưu
    savepath = img_path+'_%.2d_%.2d'%(win_size,edge_geo_dist_thresh)+'.graph_res'
    
    # Kiểm tra nếu file đã tồn tại thì bỏ qua luôn
    if os.path.exists(savepath):
        print(f'Graph already exists for {img_path}, skipping generation.')
        return 
    # ---------------------------
    if 'srns' not in edge_type:
        raise NotImplementedError
    
    # find local maxima
    vesselness = fg_prob_map
    
    im_y = vesselness.shape[0]
    im_x = vesselness.shape[1]
    y_quan = range(0,im_y,win_size)
    y_quan = sorted(list(set(y_quan) | set([im_y])))
    x_quan = range(0,im_x,win_size)
    x_quan = sorted(list(set(x_quan) | set([im_x])))
    
    max_val = []
    max_pos = []
    for y_idx in range(len(y_quan)-1):
        for x_idx in range(len(x_quan)-1):
            cur_patch = vesselness[y_quan[y_idx]:y_quan[y_idx+1],x_quan[x_idx]:x_quan[x_idx+1]]
            if np.sum(cur_patch)==0:
                max_val.append(0)
                max_pos.append((y_quan[y_idx]+cur_patch.shape[0]/2,x_quan[x_idx]+cur_patch.shape[1]/2))
            else:
                max_val.append(np.amax(cur_patch))
                temp = np.unravel_index(cur_patch.argmax(), cur_patch.shape)
                max_pos.append((y_quan[y_idx]+temp[0],x_quan[x_idx]+temp[1]))
    
    graph = nx.Graph()
            
    # add nodes
    for node_idx, (node_y, node_x) in enumerate(max_pos):
        graph.add_node(node_idx, kind='MP', y=node_y, x=node_x, label=node_idx)
        # print('node label', node_idx, 'pos', (node_y,node_x), 'added')

    speed = vesselness

    node_list = list(graph.nodes)
    for i, n in enumerate(tqdm(node_list, desc=os.path.basename(img_path), leave=False)): 
            
        phi = np.ones_like(speed)
        
        # --- SỬA LỖI TẠI ĐÂY: graph.node -> graph.nodes ---
        phi[int(graph.nodes[n]['y']), int(graph.nodes[n]['x'])] = -1
        
        if speed[int(graph.nodes[n]['y']), int(graph.nodes[n]['x'])] == 0:
            continue

        y_n = int(graph.nodes[n]['y'])
        x_n = int(graph.nodes[n]['x'])
        neighbor = speed[max(0, y_n-1):min(im_y, y_n+2), 
                         max(0, x_n-1):min(im_x, x_n+2)]
        # --------------------------------------------------
        # print('neighbor mean speed:', np.mean(neighbor))
        if np.mean(neighbor)<0.1:

            continue
               
        tt = skfmm.travel_time(phi, speed, narrow=edge_geo_dist_thresh) # travel time

        for n_comp in node_list[i+1:]:
            # --- SỬA LỖI TẠI ĐÂY NỮA ---
            y_comp = int(graph.nodes[n_comp]['y'])
            x_comp = int(graph.nodes[n_comp]['x'])
            geo_dist = tt[y_comp, x_comp] # travel time
            # ---------------------------
            
            if geo_dist < edge_geo_dist_thresh:
                graph.add_edge(n, n_comp, weight=edge_geo_dist_thresh/(edge_geo_dist_thresh+geo_dist))
                # print('An edge BTWN', 'node', n, '&', n_comp, 'is constructed')
     
    # save as a file
    nx.write_gpickle(graph, savepath, protocol=pkl.HIGHEST_PROTOCOL)
    graph.clear()
    print('generated a graph for '+img_path)


if __name__ == '__main__':

    args = parse_args()
    
    print('Called with args:')
    print(args)
    
    if args.dataset=='DRIVE':
        im_root_path = '../DRIVE/all'
        test_set_txt_path = cfg.TEST.DRIVE_SET_TXT_PATH
        im_ext = '_image.tif'
        label_ext = '_label.gif'
    elif args.dataset=='STARE':
        im_root_path = '../STARE/all'
        test_set_txt_path = cfg.TEST.STARE_SET_TXT_PATH
        im_ext = '.ppm'
        label_ext = '.ah.ppm'
    elif args.dataset=='CHASE_DB1':
        im_root_path = '../CHASE_DB1/all'
        test_set_txt_path = cfg.TEST.CHASE_DB1_SET_TXT_PATH
        im_ext = '.jpg'
        label_ext = '_1stHO.png'
        
    if args.use_multiprocessing:    
        pool = multiprocessing.Pool(processes=args.multiprocessing_num_proc)
    
    res_save_path = args.save_root + '/' + cfg.TEST.RES_SAVE_PATH if len(args.save_root)>0 else cfg.TEST.RES_SAVE_PATH
        
    if len(args.save_root)>0 and not os.path.isdir(args.save_root):
        os.mkdir(args.save_root)
    if not os.path.isdir(res_save_path):   
        os.mkdir(res_save_path)
    
    with open(test_set_txt_path) as f:
        test_img_names = [x.strip() for x in f.readlines()]
        
    len_test = len(test_img_names)
    
    data_layer_test = util.DataLayer(test_img_names, \
                                     is_training=False, \
                                     use_padding=True)
    
    network = vessel_segm_vgn(args, None)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  
    sess = tf.compat.v1.InteractiveSession(config=config)
    
    saver = tf.compat.v1.train.Saver()
    
    sess.run(tf.compat.v1.global_variables_initializer())
    # ==============================================================================
    # [START] DEEP INSPECTION: KIỂM TRA TOÀN BỘ LAYER
    # ==============================================================================
    print("\n" + "="*80)
    print(">>> BẮT ĐẦU KIỂM TRA SÂU QUÁ TRÌNH LOAD MODEL")
    print("="*80)

    # 1. Lấy danh sách tất cả biến cần train và biến global (để check cả batch norm, step...)
    # Lọc bỏ các biến của Optimizer (Adam, Momentum...) vì khi test không cần load chúng
    all_vars = [v for v in tf.compat.v1.global_variables() if 'Optimizer' not in v.name and 'Adam' not in v.name and 'Momentum' not in v.name]
    
    # 2. Lưu giá trị trước khi load (Snapshot Before)
    print("... Đang chụp trạng thái biến TRƯỚC khi load...")
    values_before = sess.run({v.name: v for v in all_vars})
    
    # 3. Load Model
    if args.model_path is not None:
        print(f">>> Đang load model từ: {args.model_path}")
        try:
            saver.restore(sess, args.model_path)
        except Exception as e:
            print(f"!!! LỖI FATAL KHI LOAD: {e}")
            exit()
    else:
        print("!!! CẢNH BÁO: Không có đường dẫn model!")

    # 4. Lưu giá trị sau khi load (Snapshot After)
    print("... Đang chụp trạng thái biến SAU khi load...")
    values_after = sess.run({v.name: v for v in all_vars})

    # 5. So sánh và Báo cáo
    loaded_vars = []
    not_loaded_vars = []
    
    print("\n" + "-"*80)
    print(f"{'TÊN BIẾN (LAYER)':<50} | {'TRẠNG THÁI':<15} | {'CHI TIẾT (Diff)'}")
    print("-"*80)

    for v in all_vars:
        name = v.name
        val_b = values_before[name]
        val_a = values_after[name]
        
        # Tính sự khác biệt tổng thể
        diff = np.sum(np.abs(val_a - val_b))
        
        # Nếu diff > 0 nghĩa là giá trị đã thay đổi -> Đã load thành công
        # (Trừ trường hợp hy hữu random ra trùng nhau, nhưng tỉ lệ gần bằng 0)
        if diff > 0.000001: 
            status = "✅ ĐÃ LOAD"
            loaded_vars.append(name)
            print(f"{name:<50} | {status:<15} | Diff: {diff:.4f}")
        else:
            # Nếu giá trị y hệt -> Chưa load (vẫn dùng giá trị khởi tạo ngẫu nhiên)
            status = "❌ CHƯA LOAD"
            not_loaded_vars.append(name)
            # In màu đỏ hoặc cảnh báo rõ
            print(f"{name:<50} | {status:<15} | !!! GIÁ TRỊ KHÔNG ĐỔI")

    print("-"*80)
    print(f">>> TỔNG KẾT:")
    print(f"   - Tổng số biến trong mạng: {len(all_vars)}")
    print(f"   - Số biến load thành công: {len(loaded_vars)}")
    print(f"   - Số biến KHÔNG load được: {len(not_loaded_vars)}")

    if len(not_loaded_vars) > 0:
        print("\n!!! CẢNH BÁO: CÁC BIẾN SAU ĐÂY ĐANG CHẠY VỚI GIÁ TRỊ NGẪU NHIÊN (RANDOM):")
        for v_name in not_loaded_vars:
            print(f"   - {v_name}")
        print(">>> GỢI Ý: Kiểm tra lại tên biến trong file model.py xem có khớp với file checkpoint không.")
        print("           Sử dụng tf.train.list_variables(args.model_path) để xem tên trong file ckpt.")
    else:
        print("\n>>> TUYỆT VỜI: TOÀN BỘ MẠNG ĐÃ ĐƯỢC LOAD ĐÚNG!")

    print("="*80 + "\n")
    # ==============================================================================
    # [END] DEEP INSPECTION
    # ==============================================================================
    if args.model_path is not None:
        print("Loading model...")
        saver.restore(sess, args.model_path)
        # --- THÊM ĐOẠN NÀY ĐỂ KIỂM TRA ---
        g_step = sess.run(network.global_step)
        print(f"DEBUG CHECK: Global Step = {g_step}")
        if g_step == 0:
            print(">>> CẢNH BÁO: Model chưa được nạp! (Global step vẫn là 0)")
        else:
            print(f">>> OK: Model đã nạp thành công từ checkpoint (Step {g_step})")
        # ---------------------------------
    
    f_log = open(os.path.join(res_save_path,'log.txt'), 'w')
    f_log.write(str(args)+'\n')
    f_log.flush()
    timer = util.Timer()
    
    print("Testing the model...")
    
    ### make cnn results ###    
    res_list = [] 
    for _ in range(int(np.ceil(float(len_test)/cfg.TRAIN.GRAPH_BATCH_SIZE))):
                        
        # get one batch
        img_list, blobs_test = data_layer_test.forward()
        
        img = blobs_test['img']
        label = blobs_test['label']
        fov = blobs_test['fov']
        
        conv_feats, fg_prob_tensor, \
        cnn_feat_dict, cnn_feat_spatial_sizes_dict = sess.run(
        [network.conv_feats,
         network.img_fg_prob,
         network.cnn_feat,
         network.cnn_feat_spatial_sizes],
        feed_dict={
            network.imgs: img,
            network.labels: label
            })
    
        cur_batch_size = len(img_list)
        for img_idx in range(cur_batch_size):
            cur_res = {}
            cur_res['img_path'] = img_list[img_idx]
            cur_res['img'] = img[[img_idx],:,:,:]
            cur_res['label'] = label[[img_idx],:,:,:]
            cur_res['conv_feats'] = conv_feats[[img_idx],:,:,:]
            cur_res['cnn_fg_prob_map'] = fg_prob_tensor[img_idx,:,:,0]
            cur_res['cnn_feat'] = {k: v[[img_idx],:,:,:] for k, v in zip(cnn_feat_dict.keys(), cnn_feat_dict.values())}
            cur_res['cnn_feat_spatial_sizes'] = cnn_feat_spatial_sizes_dict
            cur_res['graph'] = None # will be filled at the next step
            cur_res['final_fg_prob_map'] = cur_res['cnn_fg_prob_map']
            cur_res['ap_list'] = []
            
            if args.dataset=='DRIVE':
                """img_name = img_list[img_idx]
                temp = img_name[util.find(img_name,'/')[-1]:]
                if args.dataset=='DRIVE':    
                    mask = skimage.io.imread(im_root_path + temp +'_mask.gif')
                else:
                    mask = skimage.io.imread(im_root_path + temp +'_mask.tif')"""
                mask = fov[img_idx,:,:,0]
                cur_res['mask'] = mask
                
                # compute the current AP
                cur_label = label[img_idx,:,:,0]
                label_roi = cur_label[mask.astype(bool)].reshape((-1))
                fg_prob_map_roi = cur_res['cnn_fg_prob_map'][mask.astype(bool)].reshape((-1))
                _, cur_cnn_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
                cur_res['ap'] = cur_cnn_ap
                cur_res['ap_list'].append(cur_cnn_ap)
            else:
                # compute the current AP
                cur_label = label[img_idx,:,:,0].reshape((-1))
                fg_prob_map = cur_res['cnn_fg_prob_map'].reshape((-1))
                _, cur_cnn_ap = util.get_auc_ap_score(cur_label, fg_prob_map)
                cur_res['ap'] = cur_cnn_ap
                cur_res['ap_list'].append(cur_cnn_ap)

            res_list.append(cur_res)
            
    ### make final results ###        
    # make graphs
    func_arg = []
    for img_idx in range(len(res_list)):
        temp_fg_prob_map = res_list[img_idx]['final_fg_prob_map']
        func_arg.append((temp_fg_prob_map, args.edge_type, args.win_size, args.edge_geo_dist_thresh, res_list[img_idx]['img_path']))
    
    print("Generating graphs...")
    if args.use_multiprocessing:    
            list(tqdm(pool.imap(make_graph_using_srns, func_arg), total=len(func_arg)))
    else:
        for x in tqdm(func_arg):
            make_graph_using_srns(x)
    
    # load graphs
    for img_idx in range(len(res_list)):
        loadpath = res_list[img_idx]['img_path']+'_%.2d_%.2d'%(args.win_size,args.edge_geo_dist_thresh)+'.graph_res'
        temp_graph = nx.read_gpickle(loadpath)
        # ==========================================================
        # [START] KIỂM TRA NGUYÊN NHÂN 1: THỨ TỰ KHÔNG GIAN (SPATIAL ORDER)
        # ==========================================================
        print(f"\n>>> DEBUG SPATIAL ORDER: {os.path.basename(res_list[img_idx]['img_path'])}")
        
        # 1. Kiểm tra số lượng node
        # Tính toán kích thước lưới kỳ vọng dựa trên ảnh và win_size
        img_h, img_w = res_list[img_idx]['img'].shape[1:3]
        expected_rows = int(np.ceil(img_h / args.win_size))
        expected_cols = int(np.ceil(img_w / args.win_size))
        expected_nodes = expected_rows * expected_cols
        
        actual_nodes = temp_graph.number_of_nodes()
        print(f"   - Ảnh input: {img_h}x{img_w}")
        print(f"   - Lưới kỳ vọng: {expected_rows}x{expected_cols} (Win_size={args.win_size})")
        print(f"   - Số Node kỳ vọng: {expected_nodes}")
        print(f"   - Số Node thực tế: {actual_nodes}")
        
        if actual_nodes != expected_nodes:
            print(f"   🔴 LỖI NGHIÊM TRỌNG: Số lượng Node không khớp! (Thừa/Thiếu {actual_nodes - expected_nodes} node)")
            print("   -> tf.reshape sẽ bị crash hoặc dồn pixel sai vị trí.")
        else:
            print("   ✅ Số lượng Node khớp.")

        # 2. Kiểm tra Trực quan (Visual Check)
        # Tạo một ảnh map để xem Node 0, 1, 2... đang nằm ở đâu trên ảnh
        # Nếu đúng: Nó phải tạo thành gradient mượt từ trên-trái xuống dưới-phải.
        node_order_map = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Mapping index
        sorted_nodes = sorted(list(temp_graph.nodes)) # Giả sử index là 0, 1, 2...
        
        is_order_correct = True
        prev_idx = -1
        
        # Chỉ kiểm tra 10 node đầu xem có liên tiếp không
        print("   - Kiểm tra tọa độ 5 node đầu tiên (Kỳ vọng: tăng dần theo hàng):")
        for i in range(min(5, len(sorted_nodes))):
            n = sorted_nodes[i]
            y, x = int(temp_graph.nodes[n]['y']), int(temp_graph.nodes[n]['x'])
            print(f"     + Node {n}: (y={y}, x={x})")
            
            # Vẽ lên map (mỗi node là một ô vuông sáng dần)
            # Giá trị pixel = index của node
            val = i / actual_nodes 
            # Vẽ to ra một chút để dễ nhìn (bằng win_size)
            y_start, x_start = max(0, y - args.win_size//2), max(0, x - args.win_size//2)
            node_order_map[y_start:y_start+args.win_size, x_start:x_start+args.win_size] = val

        # Lưu ảnh debug ra để bạn xem
        debug_path = f"debug_spatial_{img_idx}.png"
        skimage.io.imsave(debug_path, (node_order_map*255).astype(np.uint8))
        print(f"   -> Đã lưu ảnh kiểm tra thứ tự tại: {debug_path}")
        print("   -> Hãy mở ảnh này. Nếu thấy các ô vuông xếp đều đặn từ trái qua phải: OK.")
        print("   -> Nếu thấy chấm đen/trắng nhảy lung tung: LỖI THỨ TỰ.")
        
        # ==========================================================
        # [END] DEBUG
        # ==========================================================
        res_list[img_idx]['graph'] = temp_graph
        
    # make final results 
    for img_idx in range(len(res_list)):
        
        cur_img = res_list[img_idx]['img']
        cur_conv_feats = res_list[img_idx]['conv_feats']
        cur_cnn_feat = res_list[img_idx]['cnn_feat']
        cur_cnn_feat_spatial_sizes = res_list[img_idx]['cnn_feat_spatial_sizes']  
        cur_graph = res_list[img_idx]['graph']
        
        cur_graph = nx.convert_node_labels_to_integers(cur_graph)
        node_byxs = util.get_node_byx_from_graph(cur_graph, [cur_graph.number_of_nodes()])
            
        if 'geo_dist_weighted' in args.edge_type:
            adj = nx.adjacency_matrix(cur_graph)
        else:
            adj = nx.adjacency_matrix(cur_graph,weight=None).astype(float)
            
        adj_norm = util.preprocess_graph_gat(adj)           
            
        cur_feed_dict = \
        {
        network.imgs: cur_img,
        network.conv_feats: cur_conv_feats,                           
        network.node_byxs: node_byxs,
        network.adj: adj_norm, 
        network.is_lr_flipped: False,
        network.is_ud_flipped: False
        }
        cur_feed_dict.update({network.cnn_feat[cur_key]: cur_cnn_feat[cur_key] for cur_key in network.cnn_feat.keys()})
        cur_feed_dict.update({network.cnn_feat_spatial_sizes[cur_key]: cur_cnn_feat_spatial_sizes[cur_key] for cur_key in network.cnn_feat_spatial_sizes.keys()})
    
        res_prob_map = sess.run(
        [network.post_cnn_img_fg_prob],
        feed_dict=cur_feed_dict)
        res_prob_map = res_prob_map[0]
        
        res_prob_map = res_prob_map.reshape((res_prob_map.shape[1], res_prob_map.shape[2]))
        
        # compute the current AP
        if args.dataset=='DRIVE':
            cur_label = res_list[img_idx]['label']
            cur_label = np.squeeze(cur_label)
            cur_mask = res_list[img_idx]['mask']
            label_roi = cur_label[cur_mask.astype(bool)].reshape((-1))
            fg_prob_map_roi = res_prob_map[cur_mask.astype(bool)].reshape((-1))
            _, cur_ap = util.get_auc_ap_score(label_roi, fg_prob_map_roi)
            res_prob_map = res_prob_map*cur_mask
        else:
            cur_label = res_list[img_idx]['label']
            cur_label = np.squeeze(cur_label)
            _, cur_ap = util.get_auc_ap_score(cur_label.reshape((-1)), res_prob_map.reshape((-1)))
            
        res_list[img_idx]['ap'] = cur_ap
        res_list[img_idx]['ap_list'].append(cur_ap)
        res_list[img_idx]['final_fg_prob_map'] = res_prob_map
        
    ### calculate performance measures ###
    all_labels = np.zeros((0,))
    all_preds = np.zeros((0,))
    for img_idx in range(len(res_list)):
        
        cur_label = res_list[img_idx]['label']
        cur_label = np.squeeze(cur_label)
        cur_pred = res_list[img_idx]['final_fg_prob_map']
        
        # save qualitative results
        img_path = res_list[img_idx]['img_path']
        temp = img_path[util.find(img_path,'/')[-1]:]
                
        temp_output = (cur_pred*255).astype(np.uint8)
        cur_save_path = res_save_path + temp + '_prob_final.png'
        skimage.io.imsave(cur_save_path, temp_output)
        
        cur_save_path = res_save_path + temp + '.npy'
        np.save(cur_save_path, cur_pred)
        
        temp_output = ((1.-cur_pred)*255).astype(np.uint8)
        cur_save_path = res_save_path + temp + '_prob_final_inv.png'
        skimage.io.imsave(cur_save_path, temp_output)
        # save qualitative results
        
        if args.dataset=='DRIVE':
            cur_mask = res_list[img_idx]['mask']
            cur_label = cur_label[cur_mask.astype(bool)]
            cur_pred = cur_pred[cur_mask.astype(bool)]
        
        all_labels = np.concatenate((all_labels,np.reshape(cur_label, (-1))))
        all_preds = np.concatenate((all_preds,np.reshape(cur_pred, (-1))))
        
        print('AP list for ' + res_list[img_idx]['img_path'] + ' : ' + str(res_list[img_idx]['ap_list']))
        f_log.write('AP list for ' + res_list[img_idx]['img_path'] + ' : ' + str(res_list[img_idx]['ap_list']) + '\n')
         
    auc_test, ap_test = util.get_auc_ap_score(all_labels, all_preds)
    all_labels_bin = np.copy(all_labels).astype(np.bool)
    all_preds_bin = all_preds>=0.5
    all_correct = all_labels_bin==all_preds_bin
    acc_test = np.mean(all_correct.astype(np.float32))

    print('test_acc: %.4f, test_auc: %.4f, test_ap: %.4f'%(acc_test, auc_test, ap_test))

    f_log.write('test_acc '+str(acc_test)+'\n')
    f_log.write('test_auc '+str(auc_test)+'\n')
    f_log.write('test_ap '+str(ap_test)+'\n')
    f_log.flush()
        
    f_log.close()
    sess.close()
    if args.use_multiprocessing:
        pool.terminate()
    print("Testing complete.")
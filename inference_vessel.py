import tensorflow as tf
import numpy as np
import skimage.io
import os

from model import vessel_segm_cnn   # file bạn gửi lên
from config import cfg              # dùng lại config
import util                         # nếu có util.py

# ====== THIẾT LẬP ======
CHECKPOINT_DIR = "DRIU_DRIVE/train"          # nơi bạn lưu checkpoint
IMG_PATH = "sample_image.png"                # ảnh bạn muốn dự đoán
SAVE_PATH = "output_pred.png"                # nơi lưu ảnh output
TARGET_H, TARGET_W = 608, 704                # phải trùng với khi train

# ====== KHỞI TẠO MODEL ======
args = type("Args", (), {})()    # fake args
args.cnn_model = "driu"
args.opt = "adam"
args.lr = 1e-2
args.lr_decay = "pc"
args.dataset = "DRIVE"
args.pretrained_model = None

# tạo model
network = vessel_segm_cnn(args, None)

# build model (chạy forward ảo)
dummy_input = tf.zeros((1, TARGET_H, TARGET_W, 3), dtype=tf.float32)
_ = network(dummy_input, training=False)

# ====== RESTORE CHECKPOINT ======
ckpt = tf.train.Checkpoint(model=network)
latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest:
    ckpt.restore(latest).expect_partial()
    print(f"✅ Restored model from: {latest}")
else:
    raise FileNotFoundError("❌ Không tìm thấy checkpoint trong " + CHECKPOINT_DIR)

# ====== TIỀN XỬ LÝ ẢNH ======
img = skimage.io.imread(IMG_PATH)
if img.ndim == 2:
    img = np.expand_dims(img, axis=-1)
    img = np.repeat(img, 3, axis=-1)
img = img.astype(np.float32)
img_tensor = tf.convert_to_tensor(img[np.newaxis, ...])  # thêm batch dim

# resize/pad cho đúng kích thước model
img_tensor = tf.image.resize_with_pad(img_tensor, TARGET_H, TARGET_W, method='bilinear')

# ====== DỰ ĐOÁN ======
_, fg_prob_map = network(img_tensor, training=False)  # fg_prob_map shape [1, H, W, 1]
fg_prob_map = tf.squeeze(fg_prob_map).numpy()

# ====== HẬU XỬ LÝ & LƯU ======
fg_prob_map = (fg_prob_map * 255).astype(np.uint8)
output_binary = (fg_prob_map >= 128).astype(np.uint8) * 255

skimage.io.imsave(SAVE_PATH.replace(".png", "_prob.png"), fg_prob_map)
skimage.io.imsave(SAVE_PATH.replace(".png", "_output.png"), output_binary)

print("✅ Saved result to:", SAVE_PATH)

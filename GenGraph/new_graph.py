#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load weights from a .npy weight-dict and run a forward pass to write <name>_prob.png.
Designed to work with weight dict like: { "conv1_1": {"weights":..., "biases":...}, ... }
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.keras.backend.set_image_data_format('channels_last')

def load_weight_dict(npy_path):
    d = np.load(npy_path, allow_pickle=True, encoding='latin1')
    # If loaded as 0-d ndarray containing a dict, unwrap
    if isinstance(d, np.ndarray) and d.shape == () and isinstance(d.item(), dict):
        d = d.item()
    if not isinstance(d, dict):
        raise ValueError("Loaded .npy is not a dict-like object.")
    return d

class VesselDRIU(Model):
    def __init__(self, cnn_model='driu'):
        super().__init__()
        reg = None  # keep simple; add regularizer if you want
        # VGG-style backbone (names must match keys in .npy)
        self.conv1_1 = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_1')
        self.conv1_2 = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv1_2')
        self.pool1 = layers.MaxPool2D(2, strides=2, padding='same', name='pool1')

        self.conv2_1 = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_1')
        self.conv2_2 = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv2_2')
        self.pool2 = layers.MaxPool2D(2, strides=2, padding='same', name='pool2')

        self.conv3_1 = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_1')
        self.conv3_2 = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_2')
        self.conv3_3 = layers.Conv2D(256, 3, padding='same', activation='relu', name='conv3_3')
        self.pool3 = layers.MaxPool2D(2, strides=2, padding='same', name='pool3')

        self.conv4_1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_1')
        self.conv4_2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_2')
        self.conv4_3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv4_3')

        self.conv5_1 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_1')
        self.conv5_2 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_2')
        self.conv5_3 = layers.Conv2D(512, 3, padding='same', activation='relu', name='conv5_3')

        # fc6/fc7/fc8 often converted to convs in FCN/DRIU
        # create conv layers sized dynamically later if weight shapes demand different filter counts
        self.fc6 = None
        self.fc7 = None
        self.fc8 = None

        # simple specialized layers (number of channels n)
        n = 16
        self.spe_1 = layers.Conv2D(n, 3, padding='same', activation='relu', name='spe_1')
        self.spe_2 = layers.Conv2D(n, 3, padding='same', activation='relu', name='spe_2')
        self.spe_3 = layers.Conv2D(n, 3, padding='same', activation='relu', name='spe_3')
        self.spe_4 = layers.Conv2D(n, 3, padding='same', activation='relu', name='spe_4')

        # upsample via Conv2DTranspose with init trivial
        self.resized_spe_2 = layers.Conv2DTranspose(n, 4, strides=2, padding='same', activation='relu', name='resized_spe_2')
        self.resized_spe_3 = layers.Conv2DTranspose(n, 8, strides=4, padding='same', activation='relu', name='resized_spe_3')
        self.resized_spe_4 = layers.Conv2DTranspose(n, 16, strides=8, padding='same', activation='relu', name='resized_spe_4')

        self.output_layer = layers.Conv2D(1, 1, padding='same', name='output')

    def call(self, imgs, training=False):
        # ---- Backbone ----
        x = self.conv1_1(imgs)
        conv1_2 = self.conv1_2(x)
        x = self.pool1(conv1_2)

        x = self.conv2_1(x)
        conv2_2 = self.conv2_2(x)
        x = self.pool2(conv2_2)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        conv3_3 = self.conv3_3(x)
        x = self.pool3(conv3_3)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        conv4_3 = self.conv4_3(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        conv5_3 = self.conv5_3(x)

        # ---- Specialized branches ----
        spe_1 = self.spe_1(conv1_2)
        spe_2 = self.spe_2(conv2_2)
        spe_3 = self.spe_3(conv3_3)
        spe_4 = self.spe_4(conv4_3)

        rspe_2 = self.resized_spe_2(spe_2)
        rspe_3 = self.resized_spe_3(spe_3)
        rspe_4 = self.resized_spe_4(spe_4)

        # ---- Fix shape mismatch: force resize to spe_1 HxW ----
        target_shape = tf.shape(spe_1)[1:3]  # H, W

        rspe_2 = tf.image.resize(rspe_2, target_shape, method='bilinear')
        rspe_3 = tf.image.resize(rspe_3, target_shape, method='bilinear')
        rspe_4 = tf.image.resize(rspe_4, target_shape, method='bilinear')

        # ---- Concat ----
        spe_concat = tf.concat([spe_1, rspe_2, rspe_3, rspe_4], axis=-1)

        # ---- Output ----
        output = self.output_layer(spe_concat)
        fg_prob = tf.sigmoid(output)
        return fg_prob



def assign_weights_from_dict(model, weight_dict):
    """
    For each key in weight_dict, if there is a layer in model with same name,
    attempt to set its weights [kernel, bias].
    """
    mapped = {}
    for name, val in weight_dict.items():
        # val expected to be dict with 'weights' and 'biases' (or 'bias')
        if not isinstance(val, dict):
            continue

        # Safe extraction without 'or' on numpy arrays
        if 'weights' in val:
            w = val['weights']
        elif 'weight' in val:
            w = val['weight']
        elif 'kernel' in val:
            w = val['kernel']
        else:
            w = None

        if 'biases' in val:
            b = val['biases']
        elif 'bias' in val:
            b = val['bias']
        else:
            b = None

        if w is None:
            continue

        # Find layer with this name
        try:
            layer = getattr(model, name)
        except AttributeError:
            # maybe layer is in model.layers by name
            layer = None
            for L in model.layers:
                if L.name == name:
                    layer = L
                    break
        if layer is None:
            # skip if no layer with this name
            mapped[name] = 'skipped (no matching layer)'
            continue

        # Special handling: reshape if necessary
        try:
            kernel_shape = layer.kernel.shape
            if list(w.shape) == list(kernel_shape):
                layer.set_weights([w, b] if b is not None else [w])
                mapped[name] = 'full'
            else:
                # try to reshape dense->conv if possible
                if w.ndim == 2 and len(kernel_shape) == 4:
                    w_conv = w.reshape((1,1) + w.shape)
                    if list(w_conv.shape) == list(kernel_shape):
                        layer.set_weights([w_conv, b] if b is not None else [w_conv])
                        mapped[name] = 'reshaped_dense->conv'
                        continue
                mapped[name] = f'skipped (shape mismatch: npy={w.shape} layer={tuple(kernel_shape)})'
        except Exception as e:
            mapped[name] = f'error: {e}'
    return mapped


def prepare_image(img_path):
    im = Image.open(img_path).convert("RGB")  # PPM/JPG/PNG đều vào được
    orig_size = im.size  # (W, H)

    arr = np.array(im).astype(np.float32) / 255.0
    inp = np.expand_dims(arr, axis=0)  # (1,H,W,3)

    return inp, orig_size


def save_prob_map(prob_map, out_path, orig_size):
    pm = prob_map[0, :, :, 0]  # (H,W)
    pm = np.clip(pm, 0, 1)

    pm = (pm * 255).astype(np.uint8)
    im = Image.fromarray(pm)
    im = im.resize(orig_size, Image.BILINEAR)
    im.save(out_path)


def main(args):
    print("Loading weights from:", args.npy)
    wdict = load_weight_dict(args.npy)
    print("Top keys (sample):", list(wdict.keys())[:20])

    # Build model
    model = VesselDRIU()
    # Build model by calling once with a dummy input (arbitrary size, e.g. 256x256)
    dummy = tf.zeros((1, args.test_h, args.test_w, 3), dtype=tf.float32)
    _ = model(dummy, training=False)

    # Attempt to assign weights
    print("Assigning weights from .npy to model layers where possible ...")
    mapping = assign_weights_from_dict(model, wdict)
    for k, v in mapping.items():
        print(f"{k}: {v}")

    # Prepare input image
    inp, orig_size = prepare_image(args.input_img)
    print("Input shape:", inp.shape, "original size:", orig_size)

    # Run forward
    prob = model(inp, training=False).numpy()  # shape (1,H,W,1)

    # Save prob
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input_img))[0]
    out_path = os.path.join(args.out_dir, base + "_prob.png")
    save_prob_map(prob, out_path, orig_size=orig_size)
    print("Wrote:", out_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npy", required=True, help="Path to weights .npy (dict-style)")
    p.add_argument("--input_img", required=True, help="Path to input image to run inference on")
    p.add_argument("--out_dir", default="results", help="Output directory")
    p.add_argument("--test_w", type=int, default=256, help="Dummy width to build model (any positive int)")
    p.add_argument("--test_h", type=int, default=256, help="Dummy height to build model (any positive int)")
    args = p.parse_args()
    main(args)

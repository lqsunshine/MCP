"""
The original code is from StegaStampG:
Invisible Hyperlinks in Physical Photographs,
Matthew Tancik, Ben Mildenhall, Ren Ng
University of California, Berkeley, CVPR2020
More details can be found here: https://github.com/tancik/StegaStamp
"""

import bchlib
import os
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import mean_squared_error as MSE
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #ignore warning
import tensorflow.compat.v1 as tf
tf.disable_eager_execution() #执行1.0
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

import argparse
import glob
from tqdm import tqdm
from pathlib import Path

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #cudnn版本冲突时使用


def ensure_3dim(img):
    if len(img.size) == 2:
        img = img.convert('RGB')
    return img

def load_model(model_path):
    # model_path = 'ckpt/encoder_imagenet/'
    sess = tf.InteractiveSession(graph=tf.Graph())
    model = tf.saved_model.load(sess, [tag_constants.SERVING], model_path)

    return model, sess



def encoder_image(model,sess,image):

    # print(model_path)
    secret = 'a'
    size_flag = False
    secret_size = 100

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs[
        'residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

    BCH_POLYNOMIAL = 137
    BCH_BITS = 5
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])

    # image = ensure_3dim(Image.open(image_path))
    if type(image) == Image.Image:
        image = np.array(image, dtype=np.float32) / 255.
    elif type(image) == np.ndarray:
        if image.shape[0]==3: #i.e., 3*H*W
            image = image.transpose(1, 2, 0) #H*W*3
            size_flag = True
        image = image.astype(np.float32)/ 255.


    feed_dict = {
        input_secret: [secret],
        input_image: [image]
    }
    with tf.Session():
        # start = time.time()
        hidden_img, residual = sess.run([output_stegastamp, output_residual], feed_dict=feed_dict)
        # end = time.time()
    hidden_img = (hidden_img[0] * 255).astype(np.uint8)
    residual = residual[0] + .5  # For visualization
    residual = (residual * 255).astype(np.uint8)

    # , Image.fromarray(residual)
    if size_flag:
        hidden_img = hidden_img.transpose(2,0,1)
    return hidden_img

if __name__ == "__main__":
    pass
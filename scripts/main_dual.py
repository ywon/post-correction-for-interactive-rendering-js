import time
import exr
import numpy as np
import os
import istarmap
import multiprocessing as mp
import tqdm
import argparse
import glob

import tensorflow as tf
keras = tf.keras
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Gpu is avaliable")
else:
    print("Not avail")
    exit()

# Custom ops
from reproject import CalShrinkage, AvgShrinkage, ReprojectVariance, OutlierRemoval, cuda_synchronize

from timing import Timing
from utils import DataLoader
from loss import *

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
# =============================================================================
# Configuration
results_dir = 'results/post-correction-js'
framesTranining = 101

input_dir = os.path.join(os.getcwd(), 'dataset')
SCENE = 'data_Dining-room-dynamic-32spp'
read_types = ['path_demodul', 'path', 'mvec', 'linearZ', 'pnFwidth', 'albedo', 'normal','variance','optix', 'ref', 'svgf', 'temporal_variance', 'bmfr', 'nbg', 'accum']
loader = DataLoader(os.path.join(input_dir, SCENE), read_types)

# these dimensions should be matched to the constants in cuda code
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
WINDOW_WIDTH = 15
BLOCK_WIDTH = 3
FRAME_NUM = 139 
LEARNING_RATE = 1e-2
ALPHA = 0.2
ALPHA_MOMENT = 0.9

# =============================================================================
# Initialization
#atrous_regression_alloc(IMG_HEIGHT, IMG_WIDTH, WINDOW_WIDTH)

# From keras documentation to make (nearly-)reproducible results
tf.keras.utils.set_random_seed(1234)

#comparison with NJS(load pre-trained network)
######################################################################################
#KERNEL_SIZE         = 15
#KERNEL_SIZE_SQR     = KERNEL_SIZE * KERNEL_SIZE
#parser = argparse.ArgumentParser()
#parser.add_argument("--checkPointDir", type=str, default='../results/pretrained_ckpt/')
#parser.add_argument("--loadEpoch", type=int, default=50)
#parser.add_argument("--kernelSizeSqr", type=int, default=KERNEL_SIZE_SQR)
#parser.add_argument("--mode", "-m", type=str, required=True) # train or test
#args, unknown = parser.parse_known_args()

# Create model
#optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#model = create_model(IMG_WIDTH, IMG_HEIGHT, WINDOW_WIDTH, BLOCK_WIDTH)
#model = variance_model(args, IMG_WIDTH, IMG_HEIGHT, WINDOW_WIDTH, BLOCK_WIDTH)
#model.summary()
#checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
#checkpoint_path = os.path.join(args.checkPointDir,"ckpt")
#######################################################################################

timing = Timing()
reprojTime = 0.0
trainRunningTime = 0.0
testRunningTime = 0.0

def modulate(illum, albedo):
    #return tf.math.multiply(illum, albedo)
    condition = tf.equal(albedo, 0.0)
    return tf.where(condition, illum, tf.math.multiply(illum, albedo))

def demodulate(c, albedo):
    #return tf.math.divide(c, tf.maximum(albedo, 0.0001))
    condition = tf.equal(albedo, 0.0)
    return tf.where(condition, c, tf.math.divide(c, tf.maximum(albedo, 0.0001)))
    
def albedo_func(diffuse, spec):
    return tf.math.add(diffuse, spec)

def luminance(rgb):
    return rgb[...,0:1] * 0.2126 + rgb[...,1:2] * 0.7152 + rgb[...,2:3] * 0.0722

imgs = {}
def add_write_img(tensor, name, frame):
    if name not in imgs:
        imgs[name] = []
    imgs[name].append((frame, tensor[0].numpy()))

def lerp(t1, t2, alpha):
    return t1 + alpha * (t2 - t1)

def get_relMSE(input, ref):
    eps = 1e-2
    num = np.square(np.subtract(input, ref))
    denom = np.mean(ref, axis=-1, keepdims=True)
    relMse = num / (denom * denom + eps)
    relMseMean = np.mean(relMse)
    return relMseMean
    
zeros1 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 1], dtype=tf.float32)
zeros2 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 2], dtype=tf.float32)
zeros3 = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)

total_time = 0
for item in loader:
    # Send data to GPU memory to simulate real use case (GPU path tracer)
    item.to_gpu()
    cuda_synchronize()
    #####################################################################################
    startTime = time.perf_counter()
    # tf.profiler.experimental.start('./logs')
    if item.index == 0:
        success = tf.zeros_like(zeros1, dtype=tf.bool)
        accum = tf.ones_like(zeros3)
        accum_modul = tf.ones_like(zeros3)
        accumlen = tf.ones_like(zeros1)
        history = tf.ones_like(zeros3)
        historylen = tf.zeros_like(zeros1)
        shrinkage = tf.ones_like(zeros3)
        outAlpha = tf.ones_like(zeros3)
        denominator = tf.zeros_like(zeros3)
        term = tf.zeros_like(zeros3)
        temporal_variance = item.variance
        var = item.variance
        path_demodul = demodulate(item.path, item.albedo)
        biased_demodul = demodulate(item.svgf, item.albedo)
        filtered_image = tf.ones_like(zeros3)
    else:
        var = item.variance
        ## Reprojection of variance
        reproj_input = [accum, accumlen]
        success, output, variance_prev = ReprojectVariance(item.mvec, reproj_input, temporal_variance, item.linearZ, prev_linearZ, item.normal, prev_normal, item.pnFwidth, IMG_HEIGHT, IMG_WIDTH)
        history, historylen = output
        
        isLargerThanOne = accumlen > 1.0
        new_variance = lerp(variance_prev, item.variance, ALPHA_MOMENT)
        temporal_variance = tf.where(isLargerThanOne, new_variance, item.variance)
        
        accumlen = tf.minimum(10, tf.where(success, historylen + 1, 1))
        
        #Outlier removal and demodulation step
        filtered_image = OutlierRemoval(item.path)
        path_demodul = demodulate(filtered_image, item.albedo)
        biased_demodul = demodulate(item.svgf, item.albedo)
        
        shrinkage, denominator, term = CalShrinkage(path_demodul, biased_demodul, temporal_variance, item.albedo, item.normal, IMG_HEIGHT, IMG_WIDTH, 3)
        outAlpha, out_pixel = AvgShrinkage(item.albedo, item.normal, shrinkage, IMG_HEIGHT, IMG_WIDTH, 11)
        accum = lerp(biased_demodul, path_demodul, outAlpha)
    
        #Get relL2
        #accum_modul = modulate(accum, item.albedo)
        #mse_ours = get_relMSE(item.accum, item.ref)
        #mse_biased = get_relMSE(item.optix, item.ref)  
    
    ## Modulate
    accum_modul = modulate(accum, item.albedo)

    # Accumulated -> history
    prev_linearZ = item.linearZ
    prev_normal = item.normal

    cuda_synchronize()
    took = time.perf_counter() - startTime
    #####################################################################################
    total_time += 1000 * took
    if item.index >= 0:
    # if True:
        add_write_img(accum_modul, 'output', item.index)
        #continue
           
    if item.index == FRAME_NUM: #until end of frame 
        break
# =============================================================================
out_dir = f"{results_dir}/{SCENE}"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Write images
def process_write(path, img):
    # non-zero
    img = np.maximum(0.0, img)
    if img.shape[-1] <= 3:
        exr.write(path, img)
    elif img.shape[-1] > 3:
        exr.write(path, img, channel_names=[f"ch{i}" for i in range(img.shape[-1])])
    else:
        raise NotImplementedError

print("Writing to file...", end="")
# Merge write images into one list with name
write_imgs = []
for name, imgs in imgs.items():
    write_imgs += [(name, i, img) for i, img in imgs]

# Write
with mp.Pool() as pool:
    # pool.starmap(process_write, )
    name_imgs = [(f"{out_dir}/{name}_{i:04d}.exr", img) for name, i, img in write_imgs]
    for _ in tqdm.tqdm(pool.istarmap(process_write, name_imgs), total=len(write_imgs)): pass
print("Done!")

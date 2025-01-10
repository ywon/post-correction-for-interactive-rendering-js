import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import subprocess
import json

def MSE(y_pred, y_true, target_dim=-1):
    # print('y_pred', y_pred.shape)
    # print('y_true', y_true.shape)
    return np.square(y_true - y_pred)

# input: [B, 3, H, W]
# output: scalar
def RelMSE(y_pred, y_true, target_dim=-1):
    # print('y_pred', y_pred.shape)
    # print('y_true', y_true.shape)
    true_mean = np.mean(y_true, axis=target_dim, keepdims=True)  # [B, 1, H, W]
    return np.square(y_true - y_pred) / (np.square(true_mean) + 1e-2)

# Definition from ANF
def SMAPE(y_pred, y_true, target_dim=-1):
    numerator = np.sum(np.abs(y_pred - y_true), axis=target_dim, keepdims=True)
    denominator = np.sum(np.abs(y_pred), axis=target_dim, keepdims=True) \
        + np.sum(np.abs(y_true), axis=target_dim, keepdims=True)
    return numerator / (denominator + 1e-2)

def tone_mapping(y):
    y = np.clip(y, 0, a_max=None)
    ymean = np.mean(y, axis=-1, keepdims=True)
    y = y / (ymean + 1) # Reinhard simple tone mapping
    y = np.power(y, 1 / 2.2)  # gamma correction
    return y

# Define the offset value for SRGB tone mapping
offset = 0.0031308

# Convert image from linear to sRGB color space
def sRGB(linear_val):
    # Check if the image is a single pixel
    if linear_val.shape == ():
        # Check if the pixel value is less than the offset
        if linear_val <= offset:
            # Convert the pixel value to sRGB color space
            return linear_val * 12.92
        else:
            # Convert the pixel value to sRGB color space
            return 1.055 * np.power(linear_val, 1.0 / 2.4) - 0.055
    else:
        # Check if the pixel value is less than the offset
        less = linear_val <= offset

        # Convert the pixel value to sRGB color space
        linear_val[less] = linear_val[less] * 12.92
        linear_val[~less] = 1.055 * np.power(linear_val[~less], 1.0 / 2.4) - 0.055

        # Return the image in sRGB color space
        return linear_val

def tone_mapping_srgb(y):
    y = np.clip(y, 0, a_max=None) # non-negative
    y = sRGB(y)
    y = np.clip(y, 0, 1.0) # FIXME: Do not CLIP?
    return y

def SSIM(y_pred, y_true, target_dim=-1):
    y_pred = tone_mapping_srgb(y_pred)
    y_true = tone_mapping_srgb(y_true)
    val, img = ssim(y_true, y_pred, data_range=1.0, channel_axis=target_dim, full=True)
    return img

def PSNR(y_pred, y_true, target_dim=-1):
    y_pred = tone_mapping_srgb(y_pred)
    y_true = tone_mapping_srgb(y_true)
    img = psnr(y_true, y_pred, data_range=1.0)
    return img 

def FLIP(y_pred, y_true):
    # python flip.py --reference reference.{exr|png} --test test.{exr|png}
    assert type(y_pred) == str
    assert type(y_true) == str
    # Call flip and get the score from the output
    p = subprocess.Popen(['python', 'scripts/flip.py', '--tone_mapper', 'REINHARD', '--reference', y_true, '--test', y_pred], stdout=subprocess.PIPE)
    p.wait()
    output = p.stdout.read().decode('utf-8')
    # Parse by finding line with "Mean"
    for line in output.split('\n'):
        if 'Mean' in line:
            return float(line.split(' ')[-1])
    return -1

# Video to video comparison
def VMAF_video(dist_video_path, ref_video_path):
    output_path = dist_video_path.replace('.mp4', '.json')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/local/lib/x86_64-linux-gnu'
    # exit()
    # -apply_trc iec61966_2_1
    subprocess.call([
        '/usr/local/bin/ffmpeg', '-hide_banner', '-loglevel', 'warning', 
        '-i', ref_video_path, '-i', dist_video_path, 
        '-lavfi', 
        f'libvmaf=feature=name=float_ssim|name=motion:log_fmt=json:log_path={output_path}', 
        '-f', 'null', '-'
    ], env=env)
    with open(output_path, 'r') as f:
        vmaf_data = json.load(f)
    
    # remove
    os.remove(output_path)

    return vmaf_data['pooled_metrics']['vmaf']

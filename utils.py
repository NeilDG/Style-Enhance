import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys
import cv2
from sklearn.feature_extraction import image
from IPython.display import display
from skimage.measure import compare_ssim
from metrics import MultiScaleSSIM
from PIL import Image
import math

from functools import reduce

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()[1:]), 1)

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def sigmoid_cross_entropy_with_logits(x, y):  
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

def process_test_model_args(arguments):

    phone = ""
    dped_dir = 'images/'
    test_subset = "small"
    iteration = "all"
    resolution = "orig"
    use_gpu = "true"

    for args in arguments:

        if args.startswith("model"):
            phone = args.split("=")[1]

        if args.startswith("dped_dir"):
            dped_dir = args.split("=")[1]

        if args.startswith("test_subset"):
            test_subset = args.split("=")[1]

        if args.startswith("iteration"):
            iteration = args.split("=")[1]

        if args.startswith("resolution"):
            resolution = args.split("=")[1]

        if args.startswith("use_gpu"):
            use_gpu = args.split("=")[1]

    if phone == "":
        print("\nPlease specify the model by running the script with the following parameter:\n")
        print("python test_model.py model={iphone,blackberry,sony,iphone_orig,blackberry_orig,sony_orig}\n")
        sys.exit()

    return phone, dped_dir, test_subset, iteration, resolution, use_gpu

def get_resolutions():

    # IMAGE_HEIGHT, IMAGE_WIDTH

    res_sizes = {}

    res_sizes["iphone"] = [1536, 2048]
    res_sizes["blackberry"] = [1560, 2080]
    res_sizes["sony"] = [1944, 2592]
    res_sizes["iPhone8"] = [3024, 4032]
    res_sizes["iPhone8_resize"] = [1512, 2016]
    res_sizes["Nova2i_resize"] = [1512, 2016]
    res_sizes["Nova2i"] = [3024, 4032]
    res_sizes["patch"] = [100, 100]
    res_sizes["patch96"] = [96, 96]
    res_sizes["patch_padded"] = [116, 116]
    res_sizes["patch_large"] = [192, 192]
    res_sizes["patch_padded_large"] = [212, 212]

    return res_sizes

def get_specified_res(res_sizes, phone, resolution):

    if resolution == "orig":
        IMAGE_HEIGHT = res_sizes[phone.split("_")[0]][0]
        IMAGE_WIDTH = res_sizes[phone.split("_")[0]][1]
    else:
        IMAGE_HEIGHT = res_sizes[resolution][0]
        IMAGE_WIDTH = res_sizes[resolution][1]

    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * 3

    return IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE
    
def align_image_pair(fixed, moving, fixed_path, moving_path, GOOD_MATCH_PERCENT = 0.15):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15

    fixed = fixed.convert('RGB')
    fixed = np.uint8(np.array(fixed))
    fixed = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)

    moving = moving.convert('RGB')
    moving = np.uint8(np.array(moving))
    moving = cv2.cvtColor(moving, cv2.COLOR_RGB2BGR)

    fixed_gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    moving_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(moving, None)
    keypoints2, descriptors2 = orb.detectAndCompute(fixed, None)
    
    #sift = cv2.xfeatures2d.SIFT_create()
    #keypoints1, descriptors1 = sift.detectAndCompute(moving_gray,None)
    #keypoints2, descriptors2 = sift.detectAndCompute(fixed_gray,None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(moving, keypoints1, fixed, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for j, match in enumerate(matches):
        points1[j, :] = keypoints1[match.queryIdx].pt
        points2[j, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if(h is None):
        h, mask = cv2.findHomography(points1, points2)
        
    
    # Use homography
    height, width, channels = fixed.shape
    #moving=np.uint8(moving)
    moving=np.asarray(moving, dtype=np.float32)
    print(h)
    image_final = cv2.warpPerspective(moving, h, (width, height))
    #image_final = cv2.warpAffine(moving, h[0:1], (width, height))
    cv2.imwrite(moving_path, image_final)
    cv2.imwrite(fixed_path, fixed)

    print("SSIM: " + str(compare_ssim(fixed, image_final, multichannel=True)))
    if(compare_ssim(fixed, image_final, multichannel=True) <= 0.5):
        print("#################################")
    print("PSNR: " + str(psnr(fixed, image_final)))
    
def images_to_patches(LQ, HQ, PATCH_HEIGHT, PATCH_WIDTH, LQ_Path, HQ_Path, filename, k, pair_thres, adj_thres):
    
    k = image_to_patches(HQ, LQ, PATCH_HEIGHT, PATCH_WIDTH, LQ_Path, HQ_Path, filename, k, pair_thres, adj_thres)
    return k
def image_to_patches(IMG1, IMG2, PATCH_HEIGHT, PATCH_WIDTH, IMG2_Path, IMG1_Path, filename, k, pair_thres, adj_thres):
    prev_patch = None
    for i in range(0,IMG1.size[1],PATCH_HEIGHT):
        for j in range(0,IMG1.size[0],PATCH_WIDTH):
            if(j + PATCH_WIDTH <= IMG1.size[0] and i + PATCH_HEIGHT <= IMG1.size[1]):
                box = (j, i, j+PATCH_WIDTH, i+PATCH_HEIGHT)
                IMG2_patch = IMG2.crop(box)
                IMG1_patch = IMG1.crop(box)

                IMG1_cv2 = IMG1_patch.convert('RGB')
                IMG1_cv2 = np.array(IMG1_cv2)
                IMG1_cv2 = cv2.cvtColor(IMG1_cv2, cv2.COLOR_BGR2RGB)
                #pair_eval = compare_ssim(np.array(IMG1_patch), np.array(IMG2_patch), multichannel=True)
                pair_eval = MultiScaleSSIM(np.expand_dims(IMG1_patch, axis=0), np.expand_dims(IMG2_patch, axis=0), max_val=255)

                if(pair_eval >= pair_thres and (prev_patch is None or (prev_patch is not None and compare_ssim(IMG1_cv2, prev_patch, multichannel=True) <= adj_thres))):
                    IMG2_patch.save(IMG2_Path + '(' + str(k) + ").jpg")
                    IMG1_patch.save(IMG1_Path + '(' + str(k) + ").jpg")
                    k = k + 1
                    prev_patch = IMG1_cv2
                        
    return k
import cv2
from skimage.util import random_noise
from skimage.transform import resize
import random
import numpy as np


def R_chanel(image_path):
    image = cv2.imread(image_path)
    image[:, :, 0] = 0
    image[:, :, 1] = 0
    image_r = resize(image, (512, 640)) * 255
    image_r = image_r.astype(np.uint8)
    cv2.imwrite(image_path[:-4]+"-R-ch.jpg", noiser(image_r))

def R_CHB_chanel(image_path):
    image = cv2.imread(image_path)
    image_r = resize(image, (512, 640)) * 255
    image_r = image_r.astype(np.uint8)
    cv2.imwrite(image_path[:-4]+"-IR.jpg", noiser(image_r[:, :, 2]))

def noiser(image):
    val = random.uniform(0.036, 0.107)
    noisy_img = random_noise(image, mode='gaussian', var=val ** 2)
    noisy_img = (255 * noisy_img).astype(np.uint8)
    return noisy_img

# def noiser (image, noise_level = 0.5):
#     h, w, c = image.shape
#     noisy_image = np.zeros(image.shape, np.uint8)
#     cv2.randn(noisy_image, 0, noise_level)
#     noisy_image = noisy_image.reshape(h, w, c)
#     noisy_image = cv2.add(image, noisy_image)
#     return noisy_image
# def noiser(image):
#     rows, cols = image.shape
#
#     val = random.uniform(0.036, 0.107) # Use constant variance (for testing).
#
#     # Full resolution
#     noise_im1 = np.zeros((rows, cols))
#     noise_im1 = random_noise(noise_im1, mode='gaussian', var=val ** 2, clip=False)
#
#     # Half resolution
#     noise_im2 = np.zeros((rows // 2, cols // 2))
#     noise_im2 = random_noise(noise_im2, mode='gaussian', var=(val * 20) ** 2, clip=False)  # Use val*2 (needs tuning...)
#     noise_im2 = resize(noise_im2, (rows, cols))  # Upscale to original image size
#
#     # Quarter resolution
#     noise_im3 = np.zeros((rows // 4, cols // 4))
#     noise_im3 = random_noise(noise_im3, mode='gaussian', var=(val * 100) ** 2, clip=False)  # Use val*4 (needs tuning...)
#     noise_im3 = resize(noise_im3, (rows, cols))  # What is the interpolation method?
#
#     noise_im = noise_im1 + noise_im2 + noise_im3  # Sum the noise in multiple resolutions (the mean of noise_im is around zero).
#     noisy_img = image + noise_im  # Add noise_im to the input image.
#     noisy_img = np.round((255 * noisy_img)).clip(0, 255).astype(np.uint8)
#
#     return noisy_img
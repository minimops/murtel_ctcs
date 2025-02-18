import cv2
import numpy as np
from scipy.stats import skew, kurtosis

"""
Image feature extraction functions to be used for image weather filtering with Svm or RF.
"""

################# feature extract functions

### using grayscale
def extract_brightness(img):
    return np.mean(img)


def extract_contrast(img):
    return np.std(img)


def extract_sharpness(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def extract_noise(img):
    gauss = cv2.GaussianBlur(img,(3,3),0)
    return np.std(img - gauss)


def extract_edge_density(img):
    edges = cv2.Canny(img, 100, 200)
    return np.sum(edges) / edges.size

### using color

def extract_color_hist(img):
    channels = cv2.split(img)
    color_stats = {}
    colors = ("r", "g", "b")

    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        mean_val = np.mean(hist)
        variance_val = np.var(hist)
        color_stats[f'{color}_mean'] = mean_val
        color_stats[f'{color}_variance'] = variance_val
        skewness_val = skew(hist)
        kurtosis_val = kurtosis(hist)
        color_stats[f'{color}_skewness'] = skewness_val
        color_stats[f'{color}_kurtosis'] = kurtosis_val

    return color_stats


def extract_features(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    color_stats = extract_color_hist(image_rgb)

    feature_vec = {
        "brightness": extract_brightness(image_gray),
        "contrast": extract_contrast(image_gray),
        "sharpness": extract_sharpness(image_gray),
        "noise": extract_noise(image_gray),
        "edge_density": extract_edge_density(image_gray),
        **color_stats
    }

    return feature_vec


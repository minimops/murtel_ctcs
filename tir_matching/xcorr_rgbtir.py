import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


"""
Script to match and plot an overlay of TIR on RGB.

Usage example:
    python xcorr_rgbtir.py data/align_test_dirs/fullmix --output_dir data/tir_overlays_fullmix
"""



float_scales_with_max_val = {}

def create_mask(image_shape):
    """
    Create a mask with specific areas masked out
    """
    height, width = image_shape[:2]
    channels = image_shape[2] if len(image_shape) > 2 else 1

    # Initialize mask with all areas valid (255)
    # Create same number of channels as template
    mask = np.ones((height, width, channels), dtype=np.uint8) * 255
    mask_color = 0  # Black for masked regions

    # Rectangle at the top-left
    cv2.rectangle(mask, (0, 0), (100, 10), mask_color, -1)

    # Rectangle at the top-right
    cv2.rectangle(mask, (width - 100, 0), (width, 10), mask_color, -1)

    # Triangle in the bottom-left
    triangle_points = np.array([
        (0, height),
        (0, int(height * 0.60)),
        (int(width * 0.37), height)
    ])
    cv2.fillPoly(mask, [triangle_points], mask_color)

    return mask



def apply_filters(image, filters):
    """
    Applies a series of filters to the input image.
    """
    processed_image = image.copy()

    if 'canny' in filters:
        processed_image = cv2.Canny(processed_image, 100, 200)

    if 'equalize' in filters:
        if len(processed_image.shape) == 2:
            processed_image = cv2.equalizeHist(processed_image)

    if 'gaussian' in filters:
        processed_image = cv2.GaussianBlur(processed_image, (5, 5), 0)

    return processed_image


def select_best_ir_variant(rgb_gray, tir_resized, mask):
    """
    Selects the best IR image variant (normal, inverted, or rotated) based on template matching.

    Args:
        rgb_gray (numpy.ndarray): Grayscale version of the RGB image.
        tir_resized (numpy.ndarray): The resized IR image.

    Returns:
        tuple: Best IR variant, its max_val, max_loc, and associated rotation angle.
    """
    # Create both normal and inverted variants of the IR image
    tir_resized_inverted = 255 - tir_resized
    tir_variants = [tir_resized, tir_resized_inverted]

    # Initialize variables to store the best match
    best_max_val = -float('inf')  # Start with a very low value
    best_max_loc = None
    best_tir_variant = None
    best_rotation_angle = 0  # Default to no rotation

    # Rotation angles to test (-1 to 1 in 0.2 steps)
    rotation_angles = np.arange(-1, 1.2, 0.2)

    # Iterate through the variants (normal and inverted)
    for tir_variant in tir_variants:
        for angle in rotation_angles:
            # Rotate the image
            h, w = tir_variant.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_tir = cv2.warpAffine(tir_variant, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

            # Perform template matching
            result = cv2.matchTemplate(rgb_gray, rotated_tir, cv2.TM_CCORR_NORMED, mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            # Update if this variant is better
            if max_val > best_max_val:
                best_max_val = max_val
                best_max_loc = max_loc
                best_tir_variant = rotated_tir
                best_rotation_angle = angle

    return best_tir_variant, best_max_val, best_max_loc, best_rotation_angle


def manually_scale_and_overlay(image_folder_rgb, image_folder_tir, scale_factor_height, scale_factor_width, save_dir,
                               filters, apply_filter_tir, apply_filter_rgb, alpha=0.5, plot_3_wide=False):
    """
    Runs through rgb and infrared folder and overlays infrared image on rgb image using template matching from cv2

    Args:
        image_folder_rgb (String): string of rgb image folder
        image_folder_tir (String): string of tir image folder
        scale_factor_height (float): scaling factor, since infrared is smaller than rgb
        scale_factor_width (float): scaling factor, since infrared is smaller than rgb
        save_dir (String): directory to safe the created images
    """
    os.makedirs(save_dir, exist_ok=True)

    sum_of_max_vals = 0
    num_of_matchings = 0
    rgb_images = sorted(os.listdir(image_folder_rgb))
    tir_images = sorted(os.listdir(image_folder_tir))

    image_index = 1

    for rgb_image, tir_image in zip(rgb_images, tir_images):
        # load images
        rgb_path = os.path.join(image_folder_rgb, rgb_image)
        tir_path = os.path.join(image_folder_tir, tir_image)

        rgb_img = cv2.imread(rgb_path)
        tir_img = cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or tir_img is None:
            print(f"Skipping {rgb_image} or {tir_image}: Unable to load.")
            continue

        # resize TIR image
        new_width = int(tir_img.shape[1] * scale_factor_width)
        new_height = int(tir_img.shape[0] * scale_factor_height)
        tir_resized = cv2.resize(tir_img, (new_width, new_height))

        rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

        if apply_filter_rgb:
            rgb_gray = apply_filters(rgb_gray, filters)

        if apply_filter_tir:
            tir_resized = apply_filters(tir_resized, filters)

        mask = create_mask(tir_resized.shape)

        best_tir_variant, max_val, max_loc, rotation_angle = select_best_ir_variant(rgb_gray, tir_resized, mask)

        sum_of_max_vals += max_val
        num_of_matchings += 1

        # size of the resized TIR image
        h, w = best_tir_variant.shape

        matched_rgb = rgb_img.copy()

        # tir to heatmap
        heatmap = cv2.applyColorMap(tir_resized, cv2.COLORMAP_JET)

        # alpha blending to overlay onto rgb
        roi = matched_rgb[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
        blended = cv2.addWeighted(roi, 1 - alpha, heatmap, alpha, 0)
        # replace the region in the RGB image with the blended result
        matched_rgb[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w] = blended



        # plot
        plt.figure(figsize=(15, 10))

        if plot_3_wide:
            # RGB image
            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Filtered RGB: {rgb_image}")
            plt.axis("off")

            # TIR image (template)
            plt.subplot(1, 3, 2)
            plt.imshow(best_tir_variant, cmap='gray')
            plt.title(f"Resized Filtered TIR (x{scale_factor_height}): {tir_image}")
            plt.axis("off")
            plt.subplot(1, 3, 3)

        # RGB with TIR image overlayed
        plt.imshow(cv2.cvtColor(matched_rgb, cv2.COLOR_BGR2RGB))
        # plt.title(f"RGB with Overlayed TIR (conf: {np.round(max_val, 3)})")
        plt.axis("off")
        plt.tight_layout()

        max_val_str = f"{max_val:.3f}"
        roi_round = f"{rotation_angle:.3f}"
        save_path = os.path.join(save_dir, f"{image_index}_conf_{max_val_str}_roi_{roi_round}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot: {save_path}")
        image_index += 1

    return sum_of_max_vals / num_of_matchings



# specify directory for in and output in this function
def build_directory_string(input_dir, output_dir=None):

    if output_dir is None:
        output_dir = os.path.join(input_dir, "xcorr")

    # change these to different preprocessing dirs if desired
    image_folder_rgb = os.path.join(input_dir, 'masked_rgb')
    # image_folder_rgb = "../data/superglue/med_snowcover/aligned"
    image_folder_tir = os.path.join(input_dir, 'masked_tir')

    return image_folder_rgb, image_folder_tir, output_dir


def main(input_dir, output_dir, plot_3_wide):
    # manually defined scaling factor
    scale_factor_height = 0.82
    scale_factor_width = 0.82

    image_folder_rgb, image_folder_tir, save_dir = build_directory_string(input_dir, output_dir=output_dir)

    apply_filter_tir = False
    apply_filter_rgb = False
    filters = ['gaussian']  # canny, equalize and gaussian

    manually_scale_and_overlay(image_folder_rgb, image_folder_tir, scale_factor_height, scale_factor_width, save_dir,
                               filters, apply_filter_tir, apply_filter_rgb, plot_3_wide=plot_3_wide)


if __name__ == "__main__":
    data_dir = "../data"

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Path to dir of both rgb and tir images.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to save overlays.")
    parser.add_argument("--plot_wide", action="store_true",
                        help="If set, wont just plot TIR overlayed on RGB, but the RGB,"
                             " the TIR and the overlay, side by side.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.plot_wide)

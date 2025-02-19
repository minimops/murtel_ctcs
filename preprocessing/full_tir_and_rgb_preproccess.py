import cv2
import numpy as np
import os
import argparse

"""
Script to compress, filter, and optionally mask TIR and RGB images.

Usage Example:
    python full_tir_and_rgb_proproccess.py \
    fullmix \
    --data_dir ../data/align_test_dirs \
    --do_rgb \
    --do_tir
"""

def invert_image(image):
    """Inverts pixel values in the image."""
    return 255 - image

def equalize_histogram(image):
    """Applies histogram equalization to the image."""
    # Convert to grayscale (commented example lines show alternative approach)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(img)

def apply_morphology(image, kernel_size=5):
    """Applies morphological close operation to reduce noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """Applies a Gaussian blur to smooth the image."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def crop_image(image, crop_bottom=0.24, crop_left=0.82):
    """Crops a portion from the bottom and left of the image."""
    height, width = image.shape[:2]
    new_height = int(height * (1 - crop_bottom))
    new_width = int(width * (1 - crop_left))
    return image[:new_height, new_width:]

def compress_images(input_dir, output_dir, max_dim=640, crop=True, quality=80):
    """
    Compresses and optionally crops images, resizing larger than max_dim.
    Saves processed images to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping invalid image: {file_name}")
            continue

        if crop:
            image = crop_image(image)

        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_dir, file_name)
        if file_name.lower().endswith((".jpg", ".jpeg")):
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(output_path, image)

        print(f"Compressed and saved: {output_path}")

def mask_image(image, mask_color=(128, 128, 128), is_tir=True):
    """Masks a region of the image, depending on whether it's TIR or RGB."""
    masked_image = image.copy()
    height, width = masked_image.shape[:2]

    if is_tir:
        triangle_points = np.array([
            (0, height),
            (0, int(height * 0.60)),
            (int(width * 0.37), height)
        ])
        cv2.fillPoly(masked_image, [triangle_points], mask_color)
    else:
        triangle_points = np.array([
            (0, height),
            (0, int(height * 0.57)),
            (int(width * 0.48), height)
        ])
        cv2.fillPoly(masked_image, [triangle_points], mask_color)

    # Small rectangle at the top-right
    cv2.rectangle(masked_image, (width - 100, 0), (width, 6), mask_color, -1)

    return masked_image

def filter_images(input_dir, output_dir, is_tir=True, invert=False, blur=True):
    """
    Applies basic filtering (inversion, histogram equalization, optional blur) to images.
    Saves filtered images to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            if image is not None:
                if is_tir:
                    if invert:
                        image = invert_image(image)
                    image = equalize_histogram(image)
                else:
                    image = equalize_histogram(image)
                    if blur:
                        image = apply_gaussian_blur(image)

                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, image)
                print(f"Filtered and saved: {output_path}")
            else:
                print(f"Could not read image: {image_path}")

def mask_images(input_dir, output_dir, is_tir=True):
    """Masks images from input_dir and saves them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        if image is not None:
            masked_image = mask_image(image, is_tir=is_tir)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, masked_image)
            print(f"Masked and saved: {output_path}")
        else:
            print(f"Could not read image: {image_path}")

def main(do_rgb=None, do_tir=None, data_dir=None, input_dir=None):
    """
    Handles the main workflow for compressing, filtering, and masking images.
    - do_rgb, do_tir: whether to process RGB or TIR sets.
    - data_dir, input_dir: directory paths for reading and saving images.
    """

    # For usage not via console: set defaults
    if do_rgb is None:
        do_rgb = False
        do_tir = True
        data_dir = "../../data/align_test_dirs"
        input_dir = "fullmix"

    input_dir = os.path.join(data_dir, input_dir)

    # WARNING: This creates many copies for different preprocessing steps.
    #          Select the ones you wish to create.

    rgb_input_dir = os.path.join(input_dir, "rgb")
    tir_input_dir = os.path.join(input_dir, "tir")

    tir_compressed_dir = os.path.join(input_dir, "compressed_tir")
    rgb_compressed_dir = os.path.join(input_dir, "compressed_rgb")

    tir_processed_dir = os.path.join(input_dir, "masked_tir")
    inv_tir_processed_dir = os.path.join(input_dir, "inv_masked_tir")
    rgb_processed_dir = os.path.join(input_dir, "masked_rgb")
    rgb_noblur_processed_dir = os.path.join(input_dir, "noblur_masked_rgb")

    tir_filtered_dir = os.path.join(input_dir, "filtered_tir")
    inv_tir_filtered_dir = os.path.join(input_dir, "inv_filtered_tir")
    rgb_filtered_dir = os.path.join(input_dir, "filtered_rgb")
    rgb_noblur_filtered_dir = os.path.join(input_dir, "noblur_filtered_rgb")

    tir_maskonly_dir = os.path.join(input_dir, "only_masked_tir")
    rgb_maskonly_dir = os.path.join(input_dir, "only_masked_rgb")

    if do_rgb:
        compress_images(rgb_input_dir, rgb_compressed_dir, max_dim=1280, crop=True, quality=100)
        filter_images(rgb_compressed_dir, rgb_filtered_dir, is_tir=False, blur=True)
        filter_images(rgb_compressed_dir, rgb_noblur_filtered_dir, is_tir=False, blur=False)
        mask_images(rgb_noblur_filtered_dir, rgb_noblur_processed_dir, is_tir=False)
        mask_images(rgb_filtered_dir, rgb_processed_dir, is_tir=False)
        # mask_images(rgb_compressed_dir, rgb_maskonly_dir, is_tir=False)

    if do_tir:
        compress_images(tir_input_dir, tir_compressed_dir, max_dim=1280, crop=True, quality=100)
        filter_images(tir_compressed_dir, tir_filtered_dir, is_tir=True)
        # filter_images(tir_compressed_dir, inv_tir_filtered_dir, is_tir=True, invert=True)
        mask_images(tir_filtered_dir, tir_processed_dir, is_tir=True)
        # mask_images(inv_tir_filtered_dir, inv_tir_processed_dir, is_tir=True)
        # mask_images(tir_compressed_dir, tir_maskonly_dir, is_tir=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to process RGB and TIR images by compressing, filtering, and masking."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Name of the input subdirectory (e.g., 'fullmix')."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/align_test_dirs",
        help="Path to the base data directory (default: data/align_test_dirs)."
    )
    parser.add_argument(
        "--do_rgb",
        action="store_true",
        default=False,
        help="Enable RGB processing (default: False)."
    )
    parser.add_argument(
        "--do_tir",
        action="store_true",
        default=False,
        help="Enable TIR processing (default: False)."
    )

    args = parser.parse_args()

    do_rgb = args.do_rgb
    do_tir = args.do_tir
    data_dir = args.data_dir
    input_dir = args.input_dir

    main(do_rgb, do_tir, data_dir, input_dir)

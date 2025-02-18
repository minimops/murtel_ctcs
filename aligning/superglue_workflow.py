import argparse
import os
import numpy as np
import cv2
from create_mask import create_mask
from superglue_funs import load_models, extract_keypoints, match_and_align, SUPERGLUE_PATH, SUPERPOINT_PATH


"""
Workflow script to align a full directory of images to the first (reference).
Usage example:
    python superglue_workflow.py data/align_test_dirs/fullmix data/fullmix_aligned
"""


def align_all_images(input_dir, output_dir, superpoint, superglue, device='cpu'):
    """
    Aligns all images in input_dir to the first image. Saves aligned outputs to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if not image_files:
        raise ValueError(f"No valid images found in directory {input_dir}")

    # For your 1280x790 approach
    h_res, w_res = 790, 1280
    mask = create_mask([h_res, w_res])

    # Reference image = first in the directory
    reference_image_path = os.path.join(input_dir, image_files[0])
    ref_kpts, ref_desc, ref_scores, ref_inp = extract_keypoints(
        reference_image_path, superpoint, device=device,
        resize=(w_res, h_res), mask=mask, exclude_border=True
    )
    # save the reference image as-is to output
    reference_image = cv2.imread(reference_image_path)
    cv2.imwrite(os.path.join(output_dir, image_files[0]), reference_image)

    # align all other images
    for image_file in image_files[1:]:
        image_path = os.path.join(input_dir, image_file)
        print(f"Aligning {image_file}...")

        aligned_image, matrix = match_and_align(
            ref_kpts, ref_desc, ref_scores, ref_inp,
            image_path, superpoint, superglue, device,
            resize=(w_res, h_res),
            mask=mask
        )

        if aligned_image is None or matrix is None:
            print(f"Skipping {image_file} due to alignment failure.")
            continue

        # save warp matrix
        file_name, _ = os.path.splitext(image_file)
        np.savetxt(os.path.join(output_dir, f"warpMat_{file_name}.txt"), matrix, fmt='%.6e')

        # save aligned images
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, aligned_image)
        print(f"Aligned {image_file} saved to {output_path}")


def main(input_dir, output_dir):
    superpoint, superglue, device = load_models(SUPERPOINT_PATH, SUPERGLUE_PATH)
    align_all_images(input_dir, output_dir, superpoint, superglue, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory of images to align.")
    parser.add_argument("output_dir", type=str, help="Directory to save aligned images.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)

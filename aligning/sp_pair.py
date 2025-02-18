import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from superglue_funs import load_models, extract_keypoints, match_and_align, SUPERGLUE_PATH, SUPERPOINT_PATH
from create_mask import create_mask, outside_mask, inside_mask


"""
Script to run SuperGlue on a pair of images for testing/visualization/glacier movement, etc.

Usage example:
    python sp_pair.py image1.jpg image2.jpg --output_dir data/sp_pair_out
"""


def plot_matching_keypoints(image1_rgb, image2_rgb, kpts1, kpts2, save_path=None):
    """
    Plot matched keypoints on side-by-side images.
    """
    h1, w1, _ = image1_rgb.shape
    h2, w2, _ = image2_rgb.shape
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = image1_rgb
    canvas[:h2, w1:w1 + w2] = image2_rgb

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.imshow(canvas)
    plt.axis('off')

    for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
        x2_shifted = x2 + w1
        plt.plot([x1, x2_shifted], [y1, y2], color='lime', linewidth=0.5)
        plt.scatter(x1, y1, color='red', s=2)
        plt.scatter(x2_shifted, y2, color='blue', s=2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved match visualization: {save_path}")
    else:
        plt.show()
    plt.close()


def main(img1_path, img2_path, out_dir):
    device = 'cpu'
    superpoint, superglue, device = load_models(SUPERPOINT_PATH, SUPERGLUE_PATH, device=device,
                                                sp_nms_radius=4, sp_keypoint_threshold=0.1, sp_max_keypoints=600,
                                                sp_remove_borders=5)

    # Create a mask if you want, e.g.
    mask = create_mask([790, 1280])  # Or outside_mask([...]) / inside_mask([...])

    # Extract keypoints for the first image
    kpts1, desc1, scores1, inp1 = extract_keypoints(
        img1_path, superpoint, device=device, resize=(1280, 790),
        mask=mask, exclude_border=True
    )
    # Extract keypoints for the second image
    kpts2, desc2, scores2, inp2 = extract_keypoints(
        img2_path, superpoint, device=device, resize=(1280, 790),
        mask=mask, exclude_border=True
    )

    # matching logic separately for side-by-side viz
    data = {
        'keypoints0': kpts1,
        'keypoints1': kpts2,
        'descriptors0': desc1,
        'descriptors1': desc2,
        'scores0': scores1,
        'scores1': scores2,
        'image0': inp1,
        'image1': inp2
    }
    for k, v in data.items():
        if hasattr(v, "to"):
            data[k] = v.to(device)

    pred = superglue(data)
    matches0 = pred['matches0'][0].cpu().numpy()
    valid = matches0 > -1
    kpts1_np = kpts1[0].cpu().numpy()
    kpts2_np = kpts2[0].cpu().numpy()
    matched_kpts1 = kpts1_np[valid]
    matched_kpts2 = kpts2_np[matches0[valid]]

    # Visualize side-by-side matches
    img1_bgr = cv2.imread(img1_path)
    img2_bgr = cv2.imread(img2_path)
    if (img1_bgr is None) or (img2_bgr is None):
        print("Could not read one or both input images.")
        return
    img1_bgr = cv2.resize(img1_bgr, (1280, 790))
    img2_bgr = cv2.resize(img2_bgr, (1280, 790))
    img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)

    os.makedirs(out_dir, exist_ok=True)
    match_vis_path = os.path.join(out_dir, "match_side_by_side.jpg")
    plot_matching_keypoints(img1_rgb, img2_rgb, matched_kpts1, matched_kpts2, save_path=match_vis_path)

    # Optionally compute transform matrix
    if len(matched_kpts1) < 3:
        print("Not enough matches to compute transform.")
        return
    matrix, _ = cv2.estimateAffine2D(matched_kpts2, matched_kpts1)
    if matrix is None:
        print("Could not compute transform for images.")
        return

    mat_file = os.path.join(out_dir, "affine_matrix.txt")
    np.savetxt(mat_file, matrix, fmt='%.6f')
    print(f"Saved transform matrix: {mat_file}")

    # warp image2 to image1 geometry
    h, w = img1_bgr.shape[:2]
    warped_img2 = cv2.warpAffine(img2_bgr, matrix, (w, h))
    aligned_dir = os.path.join(out_dir, "aligned")
    os.makedirs(aligned_dir, exist_ok=True)

    im1_save_path = os.path.join(aligned_dir, "image1.jpg")
    cv2.imwrite(im1_save_path, img1_bgr)
    im2_save_path = os.path.join(aligned_dir, "image2_aligned.jpg")
    cv2.imwrite(im2_save_path, warped_img2)

    print(f"Saved reference image: {im1_save_path}")
    print(f"Saved warped image: {im2_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_1_path", type=str, help="Path to first image.")
    parser.add_argument("image_2_path", type=str, help="Path to second image.")
    parser.add_argument("--output_dir", type=str, default="data/superglue/pair_test", help="Output directory.")
    args = parser.parse_args()

    main(args.image_1_path, args.image_2_path, args.output_dir)

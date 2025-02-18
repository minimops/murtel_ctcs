import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

from create_mask import outside_mask
from transfo_analyze import parse_affine_matrix


"""
Script used to create quiver vizualistions of rock galcier movement estimation.
"""


def transform_points(pts, M):
    """
    pts: np.array of shape (N, 2), containing (x, y) in each row
    M: affine transform 2x3
    Returns transformed points of shape (N, 2)
    """
    # pts to homogeneous coordinates
    ones = np.ones((pts.shape[0], 1))
    pts_hom = np.hstack([pts, ones])  # shape (N, 3)

    # apply M (2x3) to get new shape (N, 2)
    transformed = pts_hom @ M.T
    return transformed


def visualize_quiver(M, output, grid_size=5, step=1.0):
    x_vals = np.arange(-grid_size, grid_size, step)
    y_vals = np.arange(-grid_size, grid_size, step)
    xx, yy = np.meshgrid(x_vals, y_vals)
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # transform
    coords_trans = transform_points(coords, M)

    # compute displacements
    u = coords_trans[:, 0] - coords[:, 0]
    v = coords_trans[:, 1] - coords[:, 1]

    plt.figure(figsize=(8, 8))
    plt.quiver(coords[:, 0], coords[:, 1], u, v, angles='xy', scale_units='xy', scale=1, color='red')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Vector Field of Affine Transformation')
    plt.savefig("data/onlyinsidetrafo_mat.png")


def visualize_quiver_on_image(image, M, output, mask=None, step=10):

    # if no mask given, use everywhere
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)  # True everywhere

    # all valid (masked) points
    ys, xs = np.where(mask)

    sample_xs = xs
    sample_ys = ys

    # coords in (x, y) format
    coords = np.column_stack([sample_xs, sample_ys])
    trans_coords = transform_points(coords, M)

    # displacement vectors
    u = coords[:, 0] - trans_coords[:, 0]
    v = coords[:, 1] - trans_coords[:, 1]

    # print("sample_xs[:10] =", sample_xs)
    # print("sample_ys[:10] =", sample_ys)
    # print("u[:10] =", u[:10])
    # print("v[:10] =", v[:10])

    # plot
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.imshow(image, cmap='gray')
    plt.quiver(
        sample_xs[::step],  # x-coords where arrow starts
        sample_ys[::step],  # y-coords where arrow starts
        u[::step],  # delta-x
        v[::step],  # delta-y
        color='red',
        angles='xy',
        scale_units='xy',
        scale=.14,
        width=0.0035,
        headwidth=3.2,
        headlength=3,
    )

    plt.savefig(output, dpi=800)


def main(mat_path, im_path, output):
    mat = parse_affine_matrix(mat_path)

    # visualize_quiver(mat, output)

    im = cv2.imread(im_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    visualize_quiver_on_image(im_rgb, mat, mask=1 - outside_mask([790, 1280]), step=1935, output=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "matrix_path",
        type=str,
        help="path of first image."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="path of second image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_glacier_movement.png",
        help="path of image to be saved."
    )

    args = parser.parse_args()

    main(args.matrix_path, args.image_path, args.output)


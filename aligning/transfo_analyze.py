import os
import re
import numpy as np


"""
Script simply used to analyze predicted transformations.
Contains helper funs to read and write trafo mat information.
"""


def parse_affine_matrix(txt_file):
    """
    Reads a 2x3 affine matrix from a text file that looks like:
        1.00e+00 8.62e-04 1.08e+00
       -8.53e-04 1.00e+00 -6.30e-01

    Returns a numpy array of shape (2,3).
    """
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    matrix_2x3 = []
    for line in lines:
        row_vals = line.strip().split()
        row_floats = [float(val) for val in row_vals]
        matrix_2x3.append(row_floats)

    matrix_2x3 = np.array(matrix_2x3, dtype=np.float64)  # shape (2, 3)
    return matrix_2x3


def decompose_affine_2x3(M):
    """
    Given a 2x3 matrix:
       [ a  b  tx ]
       [ c  d  ty ]

    We interpret:
      - translation = (tx, ty)
      - 'best-fit' pure rotation R is derived via SVD
      - warp = ||A - R||_F, i.e. how far A is from a pure rotation.

    Returns (tx, ty, rotation_deg, warp).
    """
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]

    # translation
    trans_x = tx
    trans_y = ty

    # linear submatrix
    A = np.array([[a, b],
                  [c, d]], dtype=np.float64)

    # best-fit pure rotation
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # rotation angle from R
    # R = [[r00, r01],
    #      [r10, r11]]
    r00, r10 = R[0, 0], R[1, 0]
    rotation_rad = np.arctan2(r10, r00)
    rotation_deg = np.degrees(rotation_rad)

    # warp measure
    warp = np.linalg.norm(A - R, ord='fro')

    return trans_x, trans_y, rotation_deg, warp


def read_and_print_transform_stats(dirs_list):
    """
    For each directory in `dirs_list`, find all files that start with "warpMat"
    and end with ".txt" using a regex. Then parse their affine matrices,
    decompose, and print out the translation, rotation, and warp.
    """
    filename_pattern = re.compile(r'^warpMat.*\.txt$')

    for dir_path in dirs_list:
        print(f"\nLooking in directory: {dir_path}")
        if not os.path.isdir(dir_path):
            print(f"Not a valid directory, skipping: {dir_path}")
            continue

        candidates = os.listdir(dir_path)
        mat_files = [f for f in candidates if filename_pattern.match(f)]

        if not mat_files:
            print("No 'warpMat...txt' files found.")
            continue

        for mat_file in sorted(mat_files):
            txt_path = os.path.join(dir_path, mat_file)
            M = parse_affine_matrix(txt_path)
            tx, ty, rot_deg, warp_val = decompose_affine_2x3(M)

            print(f"File: {mat_file}")
            print(f"  X translation:      {tx:.4f}")
            print(f"  Y translation:      {ty:.4f}")
            print(f"  Rotation (deg):     {rot_deg:.4f}")
            print(f"  Warp (F-norm):      {warp_val:.4f}")
            print("-" * 45)


if __name__ == "__main__":
    # example usage:
    directories = [
        "../data/superglue/almost_max_snowcover_single_day/aligned",
        "../data/superglue/fullmix/aligned",
        "../data/superglue/full_snow_cover/aligned",
        "../data/superglue/med_snowcover/aligned",
        "../data/superglue/summer_and_first_snow/aligned",
        "../data/superglue/tir_15_days_later_than_rgb/aligned"
    ]

    read_and_print_transform_stats(directories)

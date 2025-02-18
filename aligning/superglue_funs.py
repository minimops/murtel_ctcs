import os
import warnings
import torch
import numpy as np
import cv2
from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from models.utils import read_image


"""
Helpers to be called by all superglue aligning scripts.
"""


# paths to the pretrained models
SUPERPOINT_PATH = "aligning/models/weights/superpoint_v1.pth"
SUPERGLUE_PATH = "aligning/models/weights/superglue_outdoor.pth"


def load_models(superpoint_path, superglue_path, device='cpu',
                sp_nms_radius=4, sp_keypoint_threshold=0.2, sp_max_keypoints=800, sp_remove_borders=10):

    # load SuperPoint
    superpoint_config = {
        'nms_radius': sp_nms_radius,
        'keypoint_threshold': sp_keypoint_threshold,
        'max_keypoints': sp_max_keypoints,
        'remove_borders': sp_remove_borders
    }
    warnings.simplefilter(action='ignore', category=FutureWarning)
    superpoint = SuperPoint(superpoint_config).to(device)
    superpoint.load_state_dict(torch.load(superpoint_path, map_location=device))
    superpoint.eval()

    # load SuperGlue
    superglue_config = {'weights': 'outdoor'}  # or 'indoor' if desired
    superglue = SuperGlue(superglue_config).to(device)
    superglue.load_state_dict(torch.load(superglue_path, map_location=device))
    superglue.eval()

    return superpoint, superglue, device


def extract_keypoints(
        image_path,
        superpoint,
        device='cpu',
        resize=(1280, 790),
        mask=None,
        exclude_border=True):
    """
    Extract keypoints & descriptors from an image using SuperPoint.
    :param image_path: Path to the input image
    :param superpoint: Loaded SuperPoint model
    :param device: 'cpu' or 'cuda'
    :param resize: (width, height) to resize images or None
    :param mask: Optional numpy mask to exclude keypoints
    :param exclude_border: If True, also dilate mask to remove border keypoints
    :return: (kpts, desc, scores, inp) as Tensors
    """
    _, inp, _ = read_image(
        image_path, device,
        resize=resize,
        rotation=0,
        resize_float=False
    )
    inp = inp.unsqueeze(0) if inp.dim() == 3 else inp  # shape: (1,1,H,W)

    # extract SP keypoints
    pred = superpoint({'image': inp})
    keypoints = pred['keypoints'][0]     # (N, 2)
    descriptors = pred['descriptors'][0] # (D, N)
    scores = pred['scores'][0]           # (N,)

    if mask is not None:
        print(f"Num pre mask removal kpts: {len(keypoints)}")

        h, w = inp.shape[2], inp.shape[3]
        # resize mask if needed
        if mask.shape[0] != h or mask.shape[1] != w:
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        if exclude_border:
            kernel = np.ones((3, 3), np.uint8)
            mask_resized = cv2.dilate(mask_resized, kernel, iterations=1)

        # keep only keypoints not in mask
        kpts_np = keypoints.detach().cpu().numpy()
        x_coords = np.round(kpts_np[:, 0]).astype(int)
        y_coords = np.round(kpts_np[:, 1]).astype(int)
        valid_x = np.clip(x_coords, 0, w - 1)
        valid_y = np.clip(y_coords, 0, h - 1)

        keep_indices = []
        for i, (xx, yy) in enumerate(zip(valid_x, valid_y)):
            if mask_resized[yy, xx] == 0:
                keep_indices.append(i)

        keep_t = torch.tensor(keep_indices, dtype=torch.long, device=device)
        keypoints = keypoints[keep_t]
        descriptors = descriptors[:, keep_t]
        scores = scores[keep_t]

        print(f"Num post mask removal kpts: {len(keypoints)}")

    keypoints = keypoints.unsqueeze(0)
    descriptors = descriptors.unsqueeze(0)
    scores = scores.unsqueeze(0)

    return keypoints, descriptors, scores, inp


def match_and_align(
        reference_kpts, reference_desc, reference_scores, reference_inp,
        target_path, superpoint, superglue, device,
        resize=(1280, 790),
        mask=None):
    """
    Align the 'target_path' image to the 'reference' image features
    using SuperGlue matches + estimateAffine2D.

    :param reference_kpts, reference_desc, reference_scores, reference_inp: reference image data
    :param target_path: path to the target image we want to align

    :param resize: (width, height) to match the reference resizing
    :param mask: optional mask for the target

    :return: (aligned_image, transform_matrix)
    """
    # extract keypoints for the target
    target_kpts, target_desc, target_scores, target_inp = extract_keypoints(
        target_path,
        superpoint,
        device=device,
        resize=resize,
        mask=mask,
        exclude_border=True
    )

    data = {
        'keypoints0': reference_kpts,
        'keypoints1': target_kpts,
        'descriptors0': reference_desc,
        'descriptors1': target_desc,
        'scores0': reference_scores,
        'scores1': target_scores,
        'image0': reference_inp,
        'image1': target_inp
    }
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)

    # run superglue
    pred = superglue(data)
    matches = pred['matches0'][0].cpu().numpy()  # shape (N,)
    valid = matches > -1

    # get matched coords
    ref_kpts_np = reference_kpts[0].cpu().numpy()
    tgt_kpts_np = target_kpts[0].cpu().numpy()

    matched_kpts1 = ref_kpts_np[valid]
    matched_kpts2 = tgt_kpts_np[matches[valid]]

    if len(matched_kpts1) < 3:
        print(f"No enough matches found for {target_path}; skipping alignment.")
        return None, None

    # estimate transform
    matrix, _ = cv2.estimateAffine2D(matched_kpts2, matched_kpts1)
    if matrix is None:
        print(f"Could not compute transformation for {target_path}")
        return None, None

    # apply transform
    original_target_bgr = cv2.imread(target_path)
    if original_target_bgr is None:
        print(f"Could not read {target_path}")
        return None, None

    aligned_image = cv2.warpAffine(original_target_bgr, matrix,
                                   (original_target_bgr.shape[1], original_target_bgr.shape[0]))
    return aligned_image, matrix

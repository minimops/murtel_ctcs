import cv2
import numpy as np

"""
Create masks as images.
Used for the superglue alignment to remove keypoints within the masked regions.
Also used for glacier movement estimation.
"""

# returns a mask image with a mask for metadata (top right) and ledge (bottom left)
def create_mask(image_shape):

    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255

    mask_color = 0

    # rectangle at the top-left
    # cv2.rectangle(mask, (0, 0), (181, 31), mask_color, -1)

    # rectangle at the top-right
    cv2.rectangle(mask, (width - 100, 0), (width, 10), mask_color, -1)

    # triangle in the bottom-left
    triangle_points = np.array([
        (0, height),
        (0, int(height * 0.64)),
        (int(width * 0.47), height)
    ])
    cv2.fillPoly(mask, [triangle_points], mask_color)

    return mask



###### masks for glacier movement #########

# mask all but top right
def inside_mask(image_shape, bigger_mask=False, input_path=None, output_path=None):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8) * 255
    mask_color = 1

    if input_path is not None:
        mask = cv2.imread(input_path)
        mask_color = (128, 128, 128)
        height, width = mask.shape[:2]

    # Rectangle at the top-right
    cv2.rectangle(mask, (width - 100, 0), (width, 10), mask_color, -1)

    if bigger_mask:
        # poly_points_middle = np.array([
        #     (0, int(height * 0.64)),
        #     (0,0),
        #     (int(width * 0.44), 0),
        #     (width, int(height * 0.35)),
        #     (width, height),
        #     (int(width * 0.45), height)
        #
        # ])
        poly_points_middle = np.array([
            (0, height),
            (0, 0),
            (int(width * 0.44), 0),
            (width, int(height * 0.35)),
            (width, height)

        ])
        cv2.fillPoly(mask, [poly_points_middle], mask_color)
    else:
        poly_points_middle = np.array([
            (0, height),
            (0, int(height * 0.02)),
            (width, int(height * 0.78)),
            (width, height)

        ])
        cv2.fillPoly(mask, [poly_points_middle], mask_color)

    if output_path is not None:
        cv2.imwrite(output_path, mask)

    return mask


# mask all but rock glacier
def outside_mask(image_shape, input_path=None, output_path=None):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8) * 255
    mask_color = 1

    if input_path is not None:
        mask = cv2.imread(input_path)
        mask_color = (128, 128, 128)
        height, width = mask.shape[:2]


    triangle_points = np.array([
        (0, height),
        (0, int(height * 0.64)),
        (int(width * 0.47), height)
    ])
    cv2.fillPoly(mask, [triangle_points], mask_color)

    poly_points_outside = np.array([
        (width, 0),
        (0, 0),
        (0, int(height * 0.02)),
        (width, int(height * 0.78))
    ])
    cv2.fillPoly(mask, [poly_points_outside], mask_color)


    if output_path is not None:
        cv2.imwrite(output_path, mask)

    return mask



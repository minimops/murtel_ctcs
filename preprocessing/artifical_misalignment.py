import argparse
import cv2
import numpy as np
import os
import random

"""
Script to artificially misalign RGB images to test perfomance of RGB alignment models.

Usage Example:
    `python preprocessing/artificial_misalignment.py data/align_test_dirs/fullmix/rgb data/art_misaligned_img/fullmix`
"""


def apply_translation(image, max_shift=30):
    """Applies a random translation to the image."""
    height, width = image.shape[:2]
    tx = random.randint(-max_shift, max_shift)  # Translation along x-axis
    ty = random.randint(-max_shift, max_shift)  # Translation along y-axis

    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height))
    return translated_image


def apply_rotation(image, max_angle=5):
    """Applies a random rotation to the image."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    angle = random.uniform(-max_angle, max_angle)  # Rotation angle

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def apply_warping(image, max_shift=30):
    """Applies a subtle random warping to the image."""
    height, width = image.shape[:2]

    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1]
    ])
    dst_points = np.float32([
        [random.randint(0, max_shift), random.randint(0, max_shift)],
        [width - 1 - random.randint(0, max_shift), random.randint(0, max_shift)],
        [random.randint(0, max_shift), height - 1 - random.randint(0, max_shift)],
        [width - 1 - random.randint(0, max_shift), height - 1 - random.randint(0, max_shift)]
    ])

    warp_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, warp_matrix, (width, height))
    return warped_image


def apply_transformations(image, transformations):
    """Applies a list of transformations to the image."""
    for transform in transformations:
        if transform == 'translation':
            image = apply_translation(image)
        elif transform == 'rotation':
            image = apply_rotation(image)
        elif transform == 'warping':
            image = apply_warping(image)
    return image


def misalign_images(input_dir, output_dir):
    """
    Applies random misalignment transformations to images.
    First thee images are tranlated, rotated, warped.
    All after have between 1 and 3 random transformed performed with random values.
    What is done to the images is save in their new filename postfix.
    """
    os.makedirs(output_dir, exist_ok=True)

    # all images from the directory
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = sorted(images)
    # keep first original
    images = images[1:]
    # but copy
    image = cv2.imread(os.path.join(input_dir, images[0]))
    output_path = os.path.join(output_dir, images[0])
    cv2.imwrite(output_path, image)
    print(f"first original copied to: {output_path}")


    # ensure one image has each transformation individually
    transformations = ['translation', 'rotation', 'warping']
    for i, transform in enumerate(transformations):
        if i < len(images) - 1:
            image_path = os.path.join(input_dir, images[i + 1])  # Start from the second image
            image = cv2.imread(image_path)
            if image is not None:
                transformed_image = apply_transformations(image, [transform])

                # filename to include transformation letter
                name, ext = os.path.splitext(images[i + 1])
                output_filename = f"{name}_{transform[0]}{ext}"
                output_path = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_path, transformed_image)
                print(f"{transform.capitalize()} applied and saved: {output_path}")
            else:
                print(f"Could not read image: {image_path}")

    # random 1-3 transformations to remaining images
    for image_name in images[len(transformations) + 1:]:
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        if image is not None:
            num_transforms = random.randint(1, 3)
            selected_transforms = random.sample(transformations, num_transforms)
            transformed_image = apply_transformations(image, selected_transforms)

            # filename prefix based on transformations
            name, ext = os.path.splitext(image_name)
            prefix = ''.join([t[0] for t in selected_transforms])
            output_filename = f"{name}_{prefix}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            cv2.imwrite(output_path, transformed_image)
            print(f"Transformations {selected_transforms} applied and saved: {output_path}")
        else:
            print(f"Could not read image: {image_path}")


def main(input_directory, output_directory):
    misalign_images(input_directory, output_directory)
    print(f"Misaligning images in {input_directory} and saving to {output_directory}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artificially misalign images from input directory and save to output directory.")

    parser.add_argument("input_directory", type=str, help="Path to the input directory.")
    parser.add_argument("output_directory", type=str, help="Path to the output directory.")

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory

    # input_directory = "data/align_test_dirs/fullmix/rgb"
    # output_directory = "data/art_misaligned_img/fullmix"

    misalign_images(input_directory, output_directory)

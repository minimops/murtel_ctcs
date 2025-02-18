import os
import cv2

def prepare_edge_image(image, low_threshold=50, high_threshold=150):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def save_edges(rgb_path, tir_path, output_dir,
               rgb_edge_name='rgb_edges.png', tir_edge_name='tir_edges.png',
               low_threshold=50, high_threshold=150):


    os.makedirs(output_dir, exist_ok=True)
    rgb_img = cv2.imread(rgb_path)
    tir_img = cv2.imread(tir_path)

    rgb_edges = prepare_edge_image(rgb_img, low_threshold, high_threshold)
    tir_edges = prepare_edge_image(tir_img, low_threshold, high_threshold)

    rgb_output_path = os.path.join(output_dir, rgb_edge_name)
    tir_output_path = os.path.join(output_dir, tir_edge_name)

    cv2.imwrite(rgb_output_path, rgb_edges)
    cv2.imwrite(tir_output_path, tir_edges)


def main():

    save_edges(
        rgb_path="mask_imgs/m201006150547805.jpg",
        tir_path="mask_tir_imgs/m201014153119130.jpg",
        output_dir="edges"
    )

if __name__ == "__main__":
    main()

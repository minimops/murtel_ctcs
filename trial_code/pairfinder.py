import os
from datetime import timedelta, datetime


def parse_image_time(rgb_image_path):
    try:
        if len(rgb_image_path) < 12:
            return None

        year = int("20" + rgb_image_path[1:3])
        month = int(rgb_image_path[3:5])
        day = int(rgb_image_path[5:7])
        hour = int(rgb_image_path[7:9])
        minute = int(rgb_image_path[9:11])
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def find_nearest_tir_image(rgb_image_path, tir_images, time_window):
    rgb_image_time = parse_image_time(rgb_image_path)
    if rgb_image_time is None:
        print(f'{rgb_image_path} has no valid image time')
        return None

    nearest_tir_image = None
    min_delta_time = timedelta(minutes=time_window + 1)

    for tir_image in tir_images:
        tir_image_time = parse_image_time(tir_image)
        if tir_image_time is None:
            print(f'{tir_image} has no valid image time')
            continue

        delta_time = abs(rgb_image_time - tir_image_time)
        if delta_time < min_delta_time:
            nearest_tir_image = tir_image
            min_delta_time = delta_time

    return nearest_tir_image


def save_paris_to_file(pairs, output_file):
    try:
        with open(output_file, 'w') as f:
            f.write(f"RGB_images;TIR_images\n")
            for rgb, tir in pairs:
                f.write(f"{rgb};{tir}\n")
        print(f"Saved Pairs in {output_file}")
    except Exception as e:
        print(f"Error saving Pairs in {output_file}: {e}")


def find_image_pairs(rgb_directory, tir_directory, time_window, with_path):
    pairs = []
    rgb_images = [file for file in os.listdir(rgb_directory) if file.lower().endswith(".jpg")]
    tir_images = [file for file in os.listdir(tir_directory) if file.lower().endswith(".csv")]

    for rgb_image in rgb_images:
        csv_image = find_nearest_tir_image(rgb_image, tir_images, time_window)
        if csv_image is not None:
            if with_path:
                pairs.append([os.path.join(rgb_directory, rgb_image), os.path.join(tir_directory, csv_image)])
            else:
                pairs.append([rgb_image, csv_image])

    return pairs


if __name__ == '__main__':
    rgb_directory = "../RGB_imgs"
    tir_directory = "../TIR_imgs"
    output_file = "image_pairs.csv"
    pairs = find_image_pairs(rgb_directory, tir_directory, 20, False)
    save_paris_to_file(pairs, output_file)

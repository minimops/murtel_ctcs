import os
import csv
from tkinter import *
from PIL import Image, ImageTk, ImageFile
from tkinter import ttk

# global vars
image_dir = "data/downloaded"  # directory containing images
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png', 'bmp'))]
current_image_idx = 0
labeled_data = {}


# allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# read labels from the CSV file
def load_labels(csv_file):
    labels = {}
    with open(csv_file, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['key']] = (row['tag'], int(row['value']))
    return labels

# display the next image and update the label
def show_next_image():
    global current_image_idx, image_files, labels, img_label, progress_label, progress_bar
    if current_image_idx < len(image_files):
        img_path = image_files[current_image_idx]
        image = Image.open(img_path)
        image.thumbnail((1000, 1000))
        img_tk = ImageTk.PhotoImage(image)

        img_label.config(image=img_tk)
        img_label.image = img_tk
        current_image_idx += 1

        progress_label.config(text=f"Progress: {current_image_idx}/{len(image_files)} labeled")
        progress_bar['value'] = (current_image_idx / len(image_files)) * 100
    else:
        print("All images labeled!")
        root.quit()

# handle key press for labeling
def label_image(event):
    global current_image_idx, labels, labeled_data
    key = event.char
    if key in labels:
        tag, value = labels[key]
        image_name = image_files[current_image_idx - 1]
        labeled_data[image_name] = (tag, value)
        print(f"Labeled {image_name} as {tag} (Value: {value})")
    else:
        print(f"Invalid key: {key}")

    show_next_image()

# save the labeled data into a CSV file
def save_labeled_data(output_csv):
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "tag", "value"])
        for image_name, (tag, value) in labeled_data.items():
            writer.writerow([image_name, tag, value])

labels = load_labels('labels.csv')

# GUI window
root = Tk()
root.title("Image Labeling")

# create a label to display images
img_label = Label(root)
img_label.pack()

# progress bar and label to show progress
progress_label = Label(root, text="Progress: 0/0 labeled")
progress_label.pack()

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

# frame to display possible labels
label_frame = LabelFrame(root, text="Possible Labels", padx=10, pady=10)
label_frame.pack(padx=10, pady=10, fill="both")

label_listbox = Listbox(label_frame, height=10, width=50, selectmode=SINGLE)
for key, (tag, value) in labels.items():
    label_listbox.insert(END, f"Key: {key} - Tag: {tag} (Value: {value})")
label_listbox.pack()


for key in labels.keys():
    root.bind(key, label_image)

show_next_image()
root.mainloop()

# save labeled data after labeling is complete
save_labeled_data('labeled_images.csv')

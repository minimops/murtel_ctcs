#!/bin/bash

# Usage: ./create_gif.sh <directory>
# Example: ./create_gif.sh "align_test_dirs/plots_test_nomorpho"

# Ensure ImageMagick is installed
if ! command -v convert &>/dev/null; then
    echo "Error: ImageMagick is not installed. Install it to run this script."
    exit 1
fi

# Directory to process (from command-line argument)
dir="$1"

# Check if directory was provided
if [[ -z "$dir" ]]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

echo "Processing directory: $dir"

# Output GIF path
output_gif="$dir/animation.gif"

# Temporary directory for cropped images
tmp_dir="$dir/tmp_cropped"
mkdir -p "$tmp_dir"

# Crop each PNG image and save to the temporary directory
for img in "$dir"/*.png; do
    cropped_img="$tmp_dir/$(basename "$img")"
    convert "$img" -trim +repage "$cropped_img"
done

# Create the GIF with a frame delay of 150 (1.5 seconds per frame), looping infinitely
convert -delay 150 -loop 0 "$tmp_dir"/*.png "$output_gif"

# Clean up temporary directory
rm -r "$tmp_dir"

if [[ $? -eq 0 ]]; then
    echo "GIF created and saved to: $output_gif"
else
    echo "Failed to create GIF for: $dir"
fi

echo "Processing complete."


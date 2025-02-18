#!/usr/bin/env bash
#
# side_by_side_labeled_gif.sh - Create a labeled side-by-side GIF comparing "raw" vs. "superglue_aligned".
#
# Usage:
#   side_by_side_labeled_gif.sh <before_dir> <after_dir> [output_gif] [delay]
#
#  <before_dir> : Directory with raw images (e.g., *.jpg, *.png).
#  <after_dir>  : Directory with superglue_aligned images (same filenames).
#  [output_gif] : (Optional) Name for the final side-by-side GIF. Default: sidebyside.gif
#  [delay]      : (Optional) Delay (in 1/100s of a second) between frames. Default: 50
#
# Requirements:
#   - ImageMagick (for the 'convert' command).

set -e

# 1) Parse arguments
BEFORE_DIR="$1"
AFTER_DIR="$2"
OUTPUT_GIF="${3:-sidebyside.gif}"  # Default name if none provided
DELAY="${4:-50}"                   # Default delay if none provided

if [ -z "$BEFORE_DIR" ] || [ -z "$AFTER_DIR" ]; then
  echo "Usage: $0 <before_dir> <after_dir> [output_gif] [delay]"
  exit 1
fi

# 2) Create a temporary directory to store the labeled side-by-side images
SIDEBYSIDE_DIR="sidebyside_temp_labeled"
mkdir -p "$SIDEBYSIDE_DIR"

echo "Merging images side by side with labels..."
# For each file in BEFORE_DIR, create a side-by-side image with its AFTER_DIR counterpart
for before_img in "$BEFORE_DIR"/*; do
  filename="$(basename "$before_img")"
  after_img="$AFTER_DIR/$filename"
  
  # Check if the corresponding 'after' file exists
  if [ -f "$after_img" ]; then
    # 3) Create a labeled side-by-side image
    #
    # Explanation:
    #  - We process the left (raw) image by:
    #     - Setting gravity to North (top) for text placement
    #     - Adding extra space at the top via "-splice 0x50"
    #     - Annotating it with "RAW"
    #  - Similarly for the right (superglue_aligned) image, labeled "SUPERGLUE_ALIGNED"
    #  - Finally, "+append" places them side by side horizontally.

    convert \
      \( "$before_img" \
         -gravity north -background white -splice 0x50 \
         -pointsize 25 -annotate +0+10 'RAW' \
      \) \
      \( "$after_img"  \
         -gravity north -background white -splice 0x50 \
         -pointsize 25 -annotate +0+10 'SUPERGLUE_ALIGNED' \
      \) \
      +append \
      "$SIDEBYSIDE_DIR/$filename"
    echo "  Created labeled side-by-side: $filename"
  else
    echo "  Warning: No matching file for $filename in $AFTER_DIR"
  fi
done

echo "Creating final GIF..."
# 4) Turn the side-by-side images into a single animated GIF
convert -delay "$DELAY" -loop 0 "$SIDEBYSIDE_DIR"/* "$OUTPUT_GIF"

echo "Done! Generated: $OUTPUT_GIF"
echo "Labeled side-by-side frames are in: $SIDEBYSIDE_DIR"

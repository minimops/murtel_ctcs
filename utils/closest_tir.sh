#!/bin/bash

# Usage: ./script.sh <rgb_dir> <tir_dir> <target_dir>

# Directories (taken from command-line arguments)
rgb_dir="$1"
tir_dir="$2"
target_dir="$3"

# Ensure target directory exists
mkdir -p "$target_dir"

# Function to extract the timestamp from the filename
extract_timestamp() {
    echo "$1" | grep -oP 'm\d{8}\d{4}'
}

# Iterate over each RGB file
for rgb_file in "$rgb_dir"/*.jpg; do
    rgb_name=$(basename "$rgb_file")
    rgb_time=$(extract_timestamp "$rgb_name")

    if [[ -z $rgb_time ]]; then
        echo "Could not extract timestamp from $rgb_name"
        continue
    fi

    closest_file=""
    closest_diff=999999999999 # Large initial difference

    # Compare with TIR files
    for tir_file in "$tir_dir"/*.jpg; do
        tir_name=$(basename "$tir_file")
        tir_time=$(extract_timestamp "$tir_name")

        if [[ -z $tir_time ]]; then
            echo "Could not extract timestamp from $tir_name"
            continue
        fi

        # Calculate time difference
        diff=$(( ${tir_time:1} - ${rgb_time:1} ))
        abs_diff=${diff#-} # Absolute value

        # Check for the closest match
        if (( abs_diff < closest_diff )); then
            closest_diff=$abs_diff
            closest_file=$tir_file
        fi
    done

    # Copy the closest match to the target directory
    if [[ -n $closest_file ]]; then
        cp "$closest_file" "$target_dir"
        echo "Copied $(basename "$closest_file") for $rgb_name"
    else
        echo "No match found for $rgb_name"
    fi
done

echo "Finished copying images to $target_dir"


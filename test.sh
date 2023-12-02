#!/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "No argument provided. Please provide a color name."
    exit 1
fi

# Run the commands with the provided argument
python3.10 main.py raytracer-files/ray-$1.txt && code raytracer-files/ray-$1.png && code $1.png
# python3.10 main.py raytracer-files/ray-$1.txt && code $1.png
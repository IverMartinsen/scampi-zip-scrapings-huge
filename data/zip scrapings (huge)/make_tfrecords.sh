#!/bin/bash

# Evaluate all the models
PATH_TO_FOLDERS='run_2024-02-08'

for folder in $(ls "$PATH_TO_FOLDERS"); do
    # check if directory
    if [ ! -d "$PATH_TO_FOLDERS/$folder" ]; then
        continue
    fi
    echo "Processing $folder"
    
    python from_jpg_to_tfrecords.py \
        --source "$PATH_TO_FOLDERS/$folder" \
        --compression "jpeg" \

done

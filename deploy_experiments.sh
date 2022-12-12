#!/bin/bash

# Copies [filename].json to /netscratch/martelleto/ultrasound/experiments/[filename]/config.json
# using a bash script to create the directory and copy the file.

# For all files in the experiments directory...
for filename in experiments/*.json; do
    mkdir -p /netscratch/martelleto/ultrasound/experiments/$(basename $filename .json)
    cp $filename /netscratch/martelleto/ultrasound/experiments/$(basename $filename .json)/config.json
done
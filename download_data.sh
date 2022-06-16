#!/bin/bash

data_root=data/BIOQIC
mkdir -p $data_root
cd $data_root

data_files=(
	phantom_raw.mat
	phantom_unwrapped.mat
	phantom_unwrapped_dejittered.mat
	phantom_raw_complex.mat
	four_target_phantom.mat
)
for f in "${data_files[@]}"
do
	wget https://bioqic-apps.charite.de/DownloadSamples?file=$f -O $f
done


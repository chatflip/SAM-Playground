#!/bin/bash
WEIGHT_DIR=weights
mkdir -p $WEIGHT_DIR
wget -P $WEIGHT_DIR https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
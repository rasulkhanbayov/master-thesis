#!/bin/bash

BASE_DIR=C:\/Users\/rasul\/Desktop\/multimodal-ssl-fpn-main\/multimodal-ssl-fpn-main

source $BASE_DIR/venv/Scripts/activate
cd $BASE_DIR


python --version

python train.py --model FPN --dataset Segmentation --data-ratio 1

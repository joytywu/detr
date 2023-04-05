#!/bin/bash

export dataset_file="coco_petct" 

export dataDir='/gpfs/fs0/data/stanford_data/petct/DETR_MIP2/FDG-PET-CT-Lesions/'

export num_classes=2

export outDir='outputs'

export resume="detr-r50_no-class-head.pth"

export experiment='pet_transforms'

export outDir_baseline=$outDir/$experiment

export epochs=50

python main_petct.py \
  --dataset_file $dataset_file \
  --coco_petct_path $dataDir \
  --output_dir $outDir_baseline \
  --resume $resume \
  --num_classes $num_classes \
  --lr 1e-5 \
  --lr_backbone 1e-6 \
  --epochs $epochs \
  --experiment $experiment 
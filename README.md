
##data structure
└── train_data
    ├── cropped_Depth
    ├── cropped_mask
    ├── cropped_RGB

## run
python smp_model_data_fuse.py --dataset_path [dataset_path] --output_dir [output_dir] --model_pth_path [pretrained model pth path] --train(or test) --epochs [number]

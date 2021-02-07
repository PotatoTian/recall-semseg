# Recall Loss for Semantic Segmentation (This repo implements the paper: Recall Loss for Semantic Segmentation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Overview](https://github.com/PotatoTian/recall-semseg/blob/main/image_demo/recall_demo.png?raw=true)

## Download Synthia dataset
- The model uses the SEQS-05 Sythia-seq collections [here](http://synthia-dataset.net/downloads/)
- Using different collections might require modifications to the dataloader. Please check the specific data structure and labels.
- Extract the zip / tar and modify the path appropriately in your `./configs/deeplab_synthia.yml`
```yaml
path: /datasets/synthia-seq/
```

## Create conda environment
```
conda env create -f requirements.yaml
conda activate recallCE
```

## Training 
- Run the training script.
```
python train.py --config ./configs/deeplab_synthia.yaml
```


## Testing
1. For testing on Sythia dataset, change  the validation split to `test`.
```yaml
val:
    split: test
```
2. Run the testing script
- Qulitative results will be saved in `runs/synthia/rgbd_synthia`
```
python test.py --config ./configs/deeplab_synthia.yaml
```

## Acknowledgments
- This work was supported by ONR grant N00014-18-1-2829.
- This code is built upon the implementation from [Pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

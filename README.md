# Joint Contrastive Learning with Infinite Possibilities
This is the implementation of '**Joint Contrastive Learning with Infinite Possibilities**' [NeurIPS 2020 spotlight]. Try using our new way of generating infinite (implicitly) number of training positive pairs within each training batch under the regime of contrastive learning! The published paper can be found at https://arxiv.org/abs/2009.14776. In comparison to conventional implementation e.g., MoCo [Kaiming He, et al], we derive a new loss to implement the contrastive learning. Part of the code is based on MoCo way of constructing the memory bank. 

## Requirements
* torch
* torchvision

## Datasets
* Download the ImageNet dataset from [http://www.image-net.org](http://www.image-net.org). 
* Then, move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). 

### Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To deploy unsupervised pre-training of a ResNet-50 model, run:
```
bash scripts/main_pretrain.sh
```

### Evaluation of linear classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights, run:
```
bash scripts/main_lincls.sh
```

### Models

One of our pre-trained ResNet-50 models can be downloaded from [ResNet-50](https://github.com/caiqi/Joint-Contrastive-Learning/releases/download/v0.1/checkpoint_0199.pth.tar).


## Citation
If you find this code or model useful for your research, please cite our paper (Cai Qi and Wang Yu contributed equally to this work):

    @inproceedings{cai2020joint,
      title={Joint Contrastive Learning with Infinite Possibilities},
      author={Cai, Qi and Wang, Yu and Pan, Yingwei and Yao, Ting and Mei, Tao},
      booktitle={Advances in Neural Information Processing Systems},
      year={2020}
    }

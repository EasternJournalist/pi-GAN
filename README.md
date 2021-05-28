# pi-GAN
A pytorch implementation of pi-GAN, for 3d-aware image synthesis, following the paper *pi-GAN: Periodic Implicit Generative Adversarial Networks for 3D-Aware Image Synthesis* [[Home page](https://marcoamonteiro.github.io/pi-GAN-website/)], Chan et al., Arxiv 2020 | [[bibtex](https://github.com/yenchenlin/awesome-NeRF/blob/main/NeRF-and-Beyond.bib#L24-L30)]

This pi-GAN implementation is partly based on the the version of [lucidrains/pi-GAN-pytorch: Implementation of π-GAN, for 3d-aware image synthesis, in Pytorch (github.com)](https://github.com/lucidrains/pi-GAN-pytorch)

## Requirements

* python 3.6+
* numpy
* pytorch 1.7+  & cuda

## How to run

1. You need to first configure paths in `config.py`

   `test_save_dir`:  where test results are saved (every 100 iterations).

   `data_dir`: containing images as training data. 

   `cache_dir`: where dataset are cached so that it will be loaded faster next time training. Otherwise it  could take an unbearably long time to read thousands of images every time. 

   `ckpt_dir`:  where the check point files are saved (every 1000 iterations).

2. To start a train, You may look into the `train.py`

   ```python
   from pi_gan import *
   from data import dataset
   
   # Use the piGAN class. Specify the GPU ids
   model = piGAN(dataset, devices_ids=[0, 1])
   
   # if you have a check point, load it here
   model.load_ckpt('[YOUR CKPT PATH]')
   
   # Train
   model.train(40000) # train for 40000 iterations
   ```

   


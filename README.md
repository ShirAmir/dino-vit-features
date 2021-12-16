# dino-vit-features
[[paper](https://arxiv.org/abs/2112.05814)] [[project page](dino-vit-features.github.io)]

Official implementation of the paper "Deep ViT Features as Dense Visual Descriptors".

![teaser](./teaser.png)


## Setup
Our code is developed in `pytorch` and requires the following modules: `tqdm, faiss, timm, matplotlib, pydensecrf, opencv`.
We recommend setting the running environment via Anaconda by running the following commands:
```
$ conda env create -f env/environment.yml
$ conda activate dino-vit-feats-env
```

## ViT Extractor
We provide a wrapper class for a ViT model to extract dense visual descriptors in `extractor.py`.
You can extract descriptors to `.pt` files using the following command:
```
python extractor.py --image_path <image_path> --output_path <output_path>
```
You can specify the pretrained model using the `--model` flag with the following options:
* `dino_vits8`, `dino_vits16`, `dino_vitb8`, `dino_vitb16` from the [DINO repo](https://github.com/facebookresearch/dino).
* `vit_small_patch8_224`, `vit_small_patch16_224`, `vit_base_patch8_224`, `vit_base_patch16_224` from the [timm repo](https://github.com/rwightman/pytorch-image-models/tree/master/timm).

You can specify the stride of patch extracting layer to increase resolution using the `--stride` flag.

## Part Co-segmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/shiramir/dino-vit-features/blob/main/part_cosegmentation.ipynb)
We provide a notebook for running on a single example in `part_cosegmentation.ipynb`. 

To run on several image sets, arrange each set in a directory, inside a data root directory:

```
<sets_root_name>
|
|_ <set1_name>
|  |
|  |_ img1.png
|  |_ img2.png
|   
|_ <set2_name>
   |
   |_ img1.png
   |_ img2.png
   |_ img3.png
...
```
The following command will produce results in the specified `<save_root_name>`:
```
python correspondences.py --root_dir <sets_root_name> --save_dir <save_root_name>
```

## Co-segmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/shiramir/dino-vit-features/blob/main/cosegmentation.ipynb)
We provide a notebook for running on a single example in `cosegmentation.ipynb`. 

To run on several image sets, arrange each set in a directory, inside a data root directory:

```
<sets_root_name>
|
|_ <set1_name>
|  |
|  |_ img1.png
|  |_ img2.png
|   
|_ <set2_name>
   |
   |_ img1.png
   |_ img2.png
   |_ img3.png
...
```
The following command will produce results in the specified `<save_root_name>`:
```
python correspondences.py --root_dir <sets_root_name> --save_dir <save_root_name>
```


## Point Correspondences [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/shiramir/dino-vit-features/blob/main/correspondences.ipynb)
We provide a notebook for running on a single example in `correpondences.ipynb`. 

To run on several image pairs, arrange each image pair in a directory, inside a data root directory:

```
<pairs_root_name>
|
|_ <pair1_name>
|  |
|  |_ img1.png
|  |_ img2.png
|   
|_ <pair2_name>
   |
   |_ img1.png
   |_ img2.png
...
```
The following command will produce results in the specified `<save_root_name>`:
```
python correspondences.py --root_dir <pairs_root_name> --save_dir <save_root_name>
```

## Citation
If you found this repository useful please consider starring ⭐ and citing :
```
@article{amir2021deep,
    author    = {Shir Amir and Yossi Gandelsman and Shai Bagon and Tali Dekel},
    title     = {Deep ViT Features as Dense Visual Descriptors},
    journal   = {arXiv preprint arXiv:2112.05814},
    year      = {2021}
}
```

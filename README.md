# Noisy-Fewshot
This repository contains the code for the following paper:

> A. Iscen, G. Tolias, Y. Avrithis, O. Chum, C. Schmid. "Graph convolutional networks for learning with few clean and many noisy labels", ECCV 2020

For the sake of simplicity, we only provide the code for GCN relevance weight computation for classification with prototypical classifiers. 

Python 3, PyTorch, [FAISS](https://github.com/facebookresearch/faiss) and [pygcn](https://github.com/tkipf/pygcn) are required.

##  Data:

### Features
Pre-computed descriptors can be downloaded from [here](http://ptak.felk.cvut.cz/personal/toliageo/share/fewshotclean/). 

This directory contains features extracted with ResNet10 (as trained here: [gidariss/FewShotWithoutForgetting](https://github.com/gidariss/FewShotWithoutForgetting)) and ResNet50 (as trained here: [facebookresearch/low-shot-shrink-hallucinate](https://github.com/facebookresearch/low-shot-shrink-hallucinate)).

ResNet50 features are postprocessed by PCA and their dimensionality is reduced to 256 as described in the paper. 

We provide ResNet10 and ResNet50 features for the ImageNet and Places-365 datasets. We also provide the features for the additional images retrieved from YFCC100M for these datasets (prefixed by yfcc*).

After downloading the features places them in the following directory:
```
>> mkdir data/features
```

### Noisy Images
As described in the paper, we retrieve the additional noisy images from the YFCC100M dataset. Please follow this [link](https://multimediacommons.wordpress.com/yfcc100m-core-dataset/features) for instructions on how to download the dataset.

We **do not** provide the raw images, but instead provide the **indices** of those images in the YFCC100M dataset. Index files can be found [here](http://ptak.felk.cvut.cz/personal/toliageo/share/fewshotclean/noisy_data/).

As an example, *yfcc_imagenet_test.txt* refers to the list of additional noisy images that were retrieved for the test set of novel classes for  Low-Shot ImageNet Benchmark. Each line contains:

<dataset_id, class_id>

e.g. 13, 795 means that the 13th image of the YFCC100M dataset was used for the 795th class of Shot ImageNet Benchmark.


##  Running the experiments:

We provide the code to run experiments with Protoype Classifier and our cleaning method. Low-Shot ImageNet Benchmark experiments can be run with the following command:

```
>> python run.py --dataset=imagenet --kshot=$KSHOT --network=$NETWORK
```
where ```$KSHOT``` is the number of clean images (1, 2, 5, 10 or 20) and ```$NETWORK``` is the ResNet model (resnet10 or resnet50pca).

Similarly, Low-Shot Places365 experiments can be run with the following command:

```
>> python run.py --dataset=places365 --kshot=$KSHOT --network=resnet10
```

Hyperparameters, such as ```gcnlambda```, were tuned in the validation set which contains different set of novel classes compared to the test set. Add the commandline argument ```--valset``` in order to run the experiments in the validation set of novel classes. 

Disclaimer: Not an official Google product


# HyPE-GT: where Graph Transformers meet Hyperbolic Positional Encodings

![pipeline](https://github.com/kushalbose92/HyPE-GT/blob/main/pipeline.png)

## Dependencies

* Pytorch 1.9.0
* CUDA 11.1
* Numpy 1.22.3
* DGL 0.9.1
* OGB 1.3.6

## Datasets

We perform experiments on the following datasets

* PATTERN
* CLUSTER
* MNIST
* CIFAR10
* ogbg-molhiv
* ogbg-ppa
* ogbg-molpcba
* ogbg-code2
* Amazon Computers
* Amazon Photo
* Coauthor CS
* Coauthor Physics

## Model Architecture

![HyPE-GT](https://github.com/kushalbose92/HyPE-GT/blob/main/hype-gt-model.png)

## Usage
 To run the codes for ZINC, PATTERN, and CLUSTER
 ```
 cd hyper-gt-framework
 
 sh run_sbms.sh
 sh run_superpixels.sh
 ```
 
 To run the codes for OGB graphs
 ```
 cd ogb_graphs
 sh run_molhiv.sh
 sh run_ppa.sh
 sh run_molpcba.sh
 sh run_code2.sh
 ```
 
 To run the deep GNNs models
 ```
 cd hype-deep-gnn
 sh run_all.sh
 ```

# HyPE-GT: where Graph Transformers meet Hyperbolic Positional Encodings

## Dependencies

* Pytorch 1.9.0
* CUDA 11.1
* Numpy 1.22.3
* DGL 0.9.1
* OGB 1.3.6

## Datasets

We perform experiments on the follwoing datasets

* ZINC
* PATTERN
* CLUSTER
* Amazon Computers
* Amazon Photo
* Coauthor CS
* Coauthor Physics

## Usage
 To run the codes for ZINC, PATTERN, and CLUSTER
 ```
 cd hyper-gt-framework
 
 sh run_zinc.sh
 sh run_sbms.sh
 ```
 
 To run the codes for ogbg-molhiv
 ```
 cd ogbg-molhiv
 sh run_molhiv.sh
 ```
 
 To run the deep GNNs models
 ```
 cd hyper-deep-gnn
 sh run_all.sh
 ```

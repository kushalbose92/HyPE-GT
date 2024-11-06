python main_superpixels_graph_classification.py --config 'MNIST_LapPE_Hyperboloid_HGCN.json' | tee mnist1.txt
python main_superpixels_graph_classification.py --config 'MNIST_LapPE_Hyperboloid_HNN.json' | tee mnist2.txt
python main_superpixels_graph_classification.py --config 'MNIST_LapPE_PoincareBall_HGCN.json' | tee mnist3.txt
python main_superpixels_graph_classification.py --config 'MNIST_LapPE_PoincareBall_HNN.json' | tee mnist4.txt
python main_superpixels_graph_classification.py --config 'MNIST_RWPE_Hyperboloid_HGCN.json' | tee mnist5.txt
python main_superpixels_graph_classification.py --config 'MNIST_RWPE_Hyperboloid_HNN.json' | tee mnist6.txt
python main_superpixels_graph_classification.py --config 'MNIST_RWPE_PoincareBall_HGCN.json' | tee mnist7.txt
python main_superpixels_graph_classification.py --config 'MNIST_RWPE_PoincareBall_HNN.json' | tee mnist8.txt


python main_superpixels_graph_classification.py --config 'CIFAR_LapPE_Hyperboloid_HGCN.json' | tee cifar1.txt
python main_superpixels_graph_classification.py --config 'CIFAR_LapPE_Hyperboloid_HNN.json' | tee cifar2.txt
python main_superpixels_graph_classification.py --config 'CIFAR_LapPE_PoincareBall_HGCN.json' | tee cifar3.txt
python main_superpixels_graph_classification.py --config 'CIFAR_LapPE_PoincareBall_HNN.json' | tee cifar4.txt
python main_superpixels_graph_classification.py --config 'CIFAR_RWPE_Hyperboloid_HGCN.json' | tee cifar5.txt
python main_superpixels_graph_classification.py --config 'CIFAR_RWPE_Hyperboloid_HNN.json' | tee cifar6.txt
python main_superpixels_graph_classification.py --config 'CIFAR_RWPE_PoincareBall_HGCN.json' | tee cifar7.txt
python main_superpixels_graph_classification.py --config '/CIFAR_RWPE_PoincareBall_HNN.json' | tee cifar8.txt

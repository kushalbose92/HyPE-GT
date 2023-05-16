# DATASET='coauthorcs'
# GNN_MODEL='gcn'

# python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 16 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"5.txt

# python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"6.txt

# python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"7.txt

# python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"8.txt


# python main.py --dataset 'cora' --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 50 --test_iter 10  --dropout 0.60 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --gnn_model 'jknet' --pe_init LapPE --device 0 --c 1.0 --seed 1000

# python main.py --dataset 'citeseer' --lr 0.01 --pos_enc_dim 32 --dim 256  --pe_layers 1 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.70 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --device 0 --c 1.0

# python main.py --dataset 'pubmed' --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --device 0 --c 1.0



# python main.py --dataset amazonphoto --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --gnn_model 'jknet' --pe_init LapPE --device 0 --c 1.0 --seed 1 --run_base True | tee out0.txt

# python main.py --dataset amazonphoto --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --gnn_model 'gcn' --pe_init LapPE --device 0 --c 1.0 --seed 1 --run_base False | tee out1.txt

# python main.py --dataset amazonphoto --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --gnn_model 'gcn' --pe_init RWPE --device 0 --c 1.0 --seed 1 --run_base False | tee out2.txt

# python main.py --dataset amazonphoto --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold PoincareBall --model HGCN --gnn_model 'gcn' --pe_init LapPE --device 0 --c 1.0 --seed 1 --run_base False | tee out3.txt

# python main.py --dataset amazonphoto --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold PoincareBall --model HGCN --gnn_model 'gcn' --pe_init RWPE --device 0 --c 1.0 --seed 1 --run_base False | tee out4.txt




# python main.py --dataset amazoncomputers --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.50 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --device 0 --c 1.0

# python main.py --dataset coauthorcs --lr 0.01 --pos_enc_dim 512 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --device 0 --c 1.0

# python main.py --dataset coauthorphysics --lr 0.01 --pos_enc_dim 512 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --manifold Hyperboloid --model HGCN --device 0 --c 1.0


# ----------------------------------------------------------------------

# Dataset Hyper-parameters

# Cora layers: 64, α`: 0.1, lr: 0.01, hidden: 64, λ: 0.5,
# dropout: 0.6, L2c : 0.01, L2d : 0.0005

# Citeseer layers: 32, α`: 0.1, lr: 0.01, hidden: 256, λ: 0.6,
# dropout: 0.7, L2c : 0.01, L2d : 0.0005

# Pubmed layers: 16, α`: 0.1, lr: 0.01, hidden: 256, λ: 0.4,
# dropout: 0.5, L2c : 0.0005, L2d : 0.0005
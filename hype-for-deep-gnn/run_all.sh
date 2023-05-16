DATASET='coauthorphysics'
GNN_MODEL='gcn'

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 2 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"2-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 4 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"4-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 8 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"8-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 16 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"16-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 32 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"32-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 128  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 64 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"64-layers"_"8.txt



python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 0 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"0.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 1 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"1.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 2 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"2.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 3 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"3.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 128 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 4 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"4.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 5 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"5.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 6 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"6.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 7 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"7.txt

python main.py --dataset $DATASET --lr 0.01 --pos_enc_dim 4 --dim 64  --pe_layers 2 --gcn_layers 128 --act relu --train_iter 500 --test_iter 10  --dropout 0.20 --w_decay 0.0005 --gnn_model $GNN_MODEL --pe_category 8 --device 0 --c 1.0 --seed 1 | tee $DATASET"_"$GNN_MODEL"_"128-layers"_"8.txt
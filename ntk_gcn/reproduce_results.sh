# change dataset to 'citeseer' or 'WebKB' for getting respective results
# results are for depth = [1,2,4,8,16,32,64,128]

# NTK for vanilla GCN
# ReLU GCN
python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 0
# Linear GCN
python train.py --dataset "cora" --gcn_linear 1 --gcn_skip 0

# NTK for Skip-PC
# ReLU H_0
python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 1 --skip_form "gcn" --relu_h0 1
# Linear H_0
python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 1 --skip_form "gcn" --relu_h0 0

# NTK for Skip-alpha
# ReLU H_0
python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 1 --skip_form "gcnii" --relu_h0 1
# Linear H_0
python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 1 --skip_form "gcnii" --relu_h0 0

# train GCN for specific architecture by changing the respective parameter
python train.py --train_gcn 1 --dataset "cora" --lr 0.01 --hidden 16  --layers 2 --gcn_linear 0 --gcn_skip 1 --skip_form "gcn" --skip_alpha 0.2 --relu_h0 1 --epochs 10000

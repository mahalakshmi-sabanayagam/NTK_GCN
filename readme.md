The code is adopted and changed according to our needs from https://github.com/tkipf/pygcn 

This github provides the implementation to reproduce the results in paper `Representation Power of Graph Convolutions : Neural Tangent Kernel Analysis' ([arXiv](https://arxiv.org/abs/2210.09809)).

* Install the required packages
`pip install -r requirement.txt`
  
* `cd ntk_gcn`

* There are three actions possible: <br>
1. To get the performance of depth=[1,2,4,8,16] on real datasets cora and citeseer using NTK, run the following script by changing the arguments accordingly <br>
* Linear/ReLU GCN -- `python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 0 --adj_norm "row_norm"` : pass "citeseer", --adj_norm as "col_norm" or "sym_norm" or "unnorm", for Linear GCN --gcn_linear 1

* Linear/ReLU Skip-PC or Skip-alpha -- `python train.py --dataset "cora" --gcn_linear 0 --gcn_skip 1 --skip_form "gcn" --adj_norm "row_norm"` : similar arguments as above and for skip alpha pass --skip_form "gcnii"

2. To get the kernels similar to the ones in the paper, 
* pass `--order_by_cls 1 --save_kernel 1` in the script and the kernel gets saved in the current working directory with the name '`dataset_norm_xxt_0_skip_form_depth.npy`. For getting dc_sbm results pass `--dataset "dc_sbm"`
* The kernels can be loaded in numpy and visualized as heatmaps

3. To train the GCN of depth d,
* pass `--train_gcn 1 --layers d --csigma 1` argument along with the above arguments. Note `csigma` should be `1` for linear GCN and `2` for ReLU GCN.


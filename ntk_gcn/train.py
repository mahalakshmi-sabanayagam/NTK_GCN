# taken from https://github.com/tkipf/pygcn and modified for our case
from __future__ import division
from __future__ import print_function

import time, os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy
from models import GCN_deep, GCN_skip

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--loss', type=str, default='nll',
                    help='mention the loss to optimize')
parser.add_argument('--self_loop', type=int, default=1,
                    help='1 for adding self loop, 0 for no self loop')
parser.add_argument('--feature_norm', type=int, default=1,
                    help='1 for adding featue normalisation, 0 for no feature normalisation')
parser.add_argument('--dataset', type=str, default='cora',
                    help='pass citeseer, WebKB')
parser.add_argument('--layers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--gcn_linear', type=int, default=1,
                    help='pass 0 to use relu non linearity')
parser.add_argument('--csigma', type=int, default=2,
                    help='relevant only for relu gcn, pass 1 for trying out relu')
parser.add_argument('--gcn_skip', type=int, default=0,
                    help='pass 1 for skip connection gcn')
parser.add_argument('--skip_seed', type=int, default=42,
                    help='pass the seed for weight init in skip')
parser.add_argument('--skip_form', type=str, default='gcn',
                    help='pass gcnii for skip-alpha')
parser.add_argument('--skip_alpha', type=float, default=0.2,
                    help='alpha for gcnii skip formulation')
parser.add_argument('--train_gcn', type=int, default=0,
                    help='pass 1 for training gcn')
parser.add_argument('--relu_h0', type=int, default=0,
                    help='pass 1 for applying relu to H_0')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cpu')
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0')

# Load data
if args.dataset == 'cora' or args.dataset == 'WebKB' or args.dataset == 'citeseer':
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset=args.dataset, self_loop=args.self_loop, feature_normalisation=args.feature_norm)
else:
    print('exiting! pass cora, citeseer or WebKB')
    exit()
print('number of train, val, test ', len(idx_train), len(idx_val), len(idx_test))
print('shape of adj, features, labels ', adj.shape, features.shape, labels.shape)

# Model and optimizer
if args.gcn_skip == False:
    print('Vanilla GCN ...')
    model = GCN_deep(nfeat=features.shape[1],
                     nhid=args.hidden,
                     nclass=labels.max().item() + 1,
                     layers=args.layers,
                     linear=args.gcn_linear,
                     sigma=args.csigma)
else:
    print('GCN with skip ...')
    model = GCN_skip(nfeat=features.shape[1],
                     nhid=args.hidden,
                     nclass=labels.max().item() + 1,
                     layers=args.layers,
                     linear=args.gcn_linear,
                     seed=args.skip_seed,
                     skip_formulation=args.skip_form,
                     alpha=args.skip_alpha,
                     sigma=args.csigma,
                     relu_h0=args.relu_h0)

optimizer = optim.SGD(model.parameters(),
                      lr=args.lr)
print(model)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # default loss
    loss = None
    labels_one_hot = None
    if args.loss == 'nll': #according to our setting
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    elif args.loss == 'mse':
        loss = torch.nn.MSELoss(reduction="sum")
        labels_one_hot = torch.nn.functional.one_hot(labels).type(torch.float32)
        loss_train = loss(output[idx_train], labels_one_hot[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    val = False
    if val:
        if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
            model.eval()
            output = model(features, adj)

        if args.loss == 'nll':
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        elif args.loss == 'mse':
            loss_val = loss(output[idx_val], labels_one_hot[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))
    else:
        if (epoch+1)  % 500 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj)
    loss_test = None
    if args.loss == 'nll':
        loss_test = F.nll_loss(output[idx_val], labels[idx_val])
    elif args.loss == 'mse':
        loss = torch.nn.MSELoss(reduction="sum")
        labels_one_hot = torch.nn.functional.one_hot(labels)
        loss_test = loss(output[idx_test], labels_one_hot[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "lr= {:.4f}".format(args.lr),
          "depth= {:.4f}".format(args.hidden),
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return acc_test.item()

def kappa_0(u):
    z = torch.zeros((u.shape), dtype=dtype).to(device)
    pi = torch.acos(z)*2
    r = (pi - torch.acos(u)) / pi
    r[r!=r] = 1.0
    return r

def kappa_1(u):
    z = torch.zeros((u.shape), dtype=dtype).to(device)
    pi = torch.acos(z) * 2
    r = (u*(pi - torch.acos(u)) + torch.sqrt(1-u*u))/pi
    r[r!=r] = 1.0
    return r

train_gcn = args.train_gcn
if train_gcn:
    # Train model
    t_total = time.time()
    acc_test = []
    for epoch in range(args.epochs):
        train(epoch)

        if (epoch+1) % 1000==0:
            acc = test()
            acc_test.append(acc)

    print('TEST ACCURACIES ', acc_test)
else:
    print("--------- NTK for GCN ----------")
    # change label to +1, -1 --> change 0 to -1
    labels[labels==0] = -1
    # store ground truth
    ground_truth = labels[idx_test]
    # change labels in test set to 0 for fair prediction
    labels[idx_test] = 0
    dtype= torch.float64

    depth_eval = [1,2] #,4,8,16,32,64,128]
    for d in depth_eval:
        print('Evaluating kernel for depth ', d)
        a = adj.to_dense()
        x = features @ features.t()
        a_norm = torch.norm(a)
        csigma = 1 #to avoid precision errors and it is not relevant for ntk as discussed in the paper

        sig = (a @ x @ a.t())
        non_linear_h0 = args.relu_h0
        if args.gcn_skip:
            if args.skip_form != "gcn":
                alpha = args.skip_alpha
            if non_linear_h0:
                p = torch.zeros((a.shape), dtype=dtype).to(device)
                diag_sig = torch.diagonal(x)
                sig_i = p + diag_sig.reshape(1, -1)
                sig_j = p + diag_sig.reshape(-1, 1)
                q = torch.sqrt(sig_i * sig_j)
                u = x/q
                E = (q * kappa_1(u)) * csigma
                E = E.float()
                sig = a @ E @ a.t()
            if args.skip_form == "gcn":
                sig_1 = sig
            else:
                if non_linear_h0:
                    sig = ((1-alpha)**2*sig +  (1-alpha)*alpha*(E @ a.t() + a @ E )  + alpha**2 * E)*csigma
                    sig_1 = E
                else:
                    sig = ((1-alpha)**2*sig +  (1-alpha)*alpha*(x @ a.t() + a @ x )  + alpha**2 * x)*csigma
                    sig_1 = x
        kernel = torch.zeros((a.shape), dtype=dtype).to(device)
        depth = d

        if args.gcn_linear == True:
            print('linear GCN....')
            # compute sigma_n + sigma_(n-1) * SS^T + ... + sigma_1 * SS^T (n-1 times)
            for i in range(depth,0,-1):
                t = torch.ones((a.shape), dtype=dtype).to(device)
                for j in range(i):
                    t = (t* (a @ a.t())) * csigma
                kernel += sig * t
                sig = (a @ sig @ a.t()) *csigma
                if args.gcn_skip:
                    #sig = sig + sig_1
                    sig = (1-alpha)**2 * sig + alpha**2 * sig_1
        else:
            print('Relu GCN....')
            # compute sigma_n + sigma_(n-1) * SS^T * der_relu_(n-1) + ... + sigma_1 * SS^T (n-1 times) * der_relu(n-1) * ... der_relu(1)
            kernel_sub = torch.zeros((depth, a.shape[0], a.shape[1]), dtype=dtype).to(device)
            for i in range(depth):
                p = torch.zeros((a.shape), dtype=dtype).to(device)
                diag_sig = torch.diagonal(sig)
                sig_i = p + diag_sig.reshape(1, -1)
                sig_j = p + diag_sig.reshape(-1, 1)
                q = torch.sqrt(sig_i * sig_j)
                u = sig/q
                E = (q * kappa_1(u)) * csigma
                E_der = (kappa_0(u)) * csigma
                kernel_der = (a @ a.t()) * E_der
                kernel_sub[i] += sig * kernel_der

                E = E.float()
                sig = a @ E @ a.t()
                if args.gcn_skip:
                    if args.skip_form == "gcn":
                        sig = sig + sig_1
                    else:
                        sig = (1-alpha)**2 * sig + alpha**2 * sig_1
                for j in range(i):
                    kernel_sub[j] *= kernel_der

            kernel += torch.sum(kernel_sub, dim=0)

        kernel += sig

        # compute derivative of sigmoid layer
        one_n = torch.ones((a.shape), dtype=dtype).to(device)
        p = torch.zeros((a.shape), dtype=dtype).to(device)
        diag_sig = torch.diagonal(sig)
        sig_i = p+diag_sig.reshape(1,-1)
        sig_j = p+diag_sig.reshape(-1,1)
        p = sig_i+sig_j
        q = sig_i*sig_j
        r = 2*sig**2
        der_new = 0.25*(one_n - 0.25*p + 0.0625*(q+r) + 0.125*(sig_i*sig_i + sig_j*sig_j))

        # compute final kernel
        kernel = kernel * der_new
        if args.gcn_linear == True:
            kernel *= csigma**(d-1)

        # compute f(x)
        id_t = idx_test[0]
        id_train = idx_train[-1]+1
        kernel_train = kernel[:id_train,:id_train]
        labels_train = labels[:id_train].type(torch.double)
        kernel_test = kernel[id_t:, :id_train]
        kernel_inv = torch.pinverse(kernel_train, rcond=1e-8)

        output = kernel_test @ kernel_inv.type(torch.float64) @ labels_train.type(torch.float64)

        # compute accuracy of the prediction
        pred = output
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        pred = pred.type_as(ground_truth)
        acc = pred.eq(ground_truth).double().sum() / len(ground_truth)
        print('Test accuracy using NTK ', acc)

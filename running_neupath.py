#!/usr/bin/env python
# coding: utf-8

import dgl
import scipy.io
from tqdm import tqdm
from NeuPath import NeuPath
import argparse
from utils import *
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(43)


def main(G, labels):
    target_type = 'paper'

    # generate train/val/test split
    # labels = torch.tensor(labels).long()
    pid = len(labels)
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0: 400]).long()
    val_idx = torch.tensor(shuffle[400:500]).long()
    test_idx = torch.tensor(shuffle[500:900]).long()

    # prepare the training, validation, testing samples
    train_labels_mat = labels[train_idx]
    train_indices = np.argsort(-train_labels_mat, axis=1)[:, :args.k]
    train_labels = torch.tensor(np.take_along_axis(train_labels_mat, train_indices, axis=1)).float()
    print("train labels print:", train_labels[10, :])
    val_labels_mat = labels[val_idx]
    val_indices = np.argsort(-val_labels_mat, axis=1)[:, :args.k]
    val_labels = torch.tensor(np.take_along_axis(val_labels_mat, val_indices, axis=1)).float()

    test_labels_mat = labels[test_idx]
    test_indices = np.argsort(-test_labels_mat, axis=1)[:, :args.k]
    test_labels = torch.tensor(np.take_along_axis(test_labels_mat, test_indices, axis=1)).float()

    G = G.to(device)
    # initialized node features and edge features : we set all node features as the same first
    G.node_dict = {}
    G.edge_dict = {}
    initial_vectors = nn.Parameter(torch.Tensor(2, args.n_inp), requires_grad=False).to(device)
    nn.init.xavier_uniform_(initial_vectors)
    equal_vec = initial_vectors[0]  # query node = target node
    inequal_vec = initial_vectors[1]  # query node != target node
    for ntype in G.ntypes:
        G.node_dict[ntype] = len(G.node_dict)
        G.nodes[ntype].data['inp'] = inequal_vec.repeat(G.number_of_nodes(ntype), 1).to(device)
    for etype in G.etypes:
        G.edge_dict[etype] = len(G.edge_dict)
        edge_emb = torch.ones(G.number_of_edges(etype), dtype=torch.long) * G.edge_dict[etype]
        G.edges[etype].data['id'] = edge_emb.to(device)

    # initialized model and set optimization operator
    model = NeuPath(G,
                    n_inp=args.n_inp,
                    n_hid=args.n_hid,
                    n_out=1,
                    n_layers=args.n_layer,
                    n_heads=args.n_head,
                    use_norm=True).to(device)
    # model = HeteroRGCN(G, in_size=args.n_inp, hidden_size=args.n_hid, out_size=1).to(device)

    print('Training NeuPath with #param: %d' % (get_n_params(model)))
    # set optimization operator
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.n_epoch, max_lr=args.max_lr)

    # model training-validation-testing
    tmp_emb = inequal_vec.repeat(G.number_of_nodes(target_type), 1)
    for epoch in np.arange(args.n_epoch) + 1:
        # ********* train *************
        l_train_loss = list()
        l_train_logit = list()
        model.train()
        for i in tqdm(range(len(train_idx))):
            idx = train_idx[i]
            G.nodes[target_type].data['inp'] = tmp_emb
            G.nodes[target_type].data['inp'][idx] = equal_vec
            logit = model(G, target_type)
            scores = np.take_along_axis(logit, train_indices[i], axis=0).reshape(-1)
            loss = F.mse_loss(scores, train_labels[i].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

            l_train_loss.append(loss.cpu().detach().item())
            l_train_logit.append(logit.cpu().detach().reshape(-1).tolist())
        # print("l_train_logit:", l_train_logit)
        scheduler.step()
        train_rmse = np.sqrt(sum(l_train_loss) / len(train_idx))
        argsorted_logits_train = np.argsort(np.array(l_train_logit), axis=1)
        train_ndcg = compute_ndcg(train_indices, argsorted_logits_train)
        print('Epoch: %d LR: %.5f Train RMSE %.4f, Train NDCG %.4f' % (
            epoch,
            optimizer.param_groups[0]['lr'],
            train_rmse,
            train_ndcg
            ))

        # ********* valid *************
        model.eval()
        l_val_loss = list()
        l_val_logit = list()
        with torch.no_grad():
            for i in tqdm(range(len(val_idx))):
                idx = val_idx[i]
                G.nodes[target_type].data['inp'] = tmp_emb
                G.nodes[target_type].data['inp'][idx] = equal_vec

                logit = model(G, target_type)
                scores = np.take_along_axis(logit, val_indices[i], axis=0).reshape(-1)
                loss = F.mse_loss(scores, val_labels[i].to(device))
                l_val_loss.append(loss.cpu().detach().item())
                l_val_logit.append(logit.cpu().detach().reshape(-1).tolist())

            val_rmse = np.sqrt(sum(l_val_loss) / len(val_idx))
            argsorted_logits_val = np.argsort(np.array(l_val_logit), axis=1)
            val_ndcg = compute_ndcg(val_indices, argsorted_logits_val)
            print('Epoch: %d LR: %.5f Val RMSE %.4f, Val NDCG %.4f' % (
                epoch,
                optimizer.param_groups[0]['lr'],
                val_rmse.item(),
                val_ndcg
            ))

    # ********* test *************
    l_test_loss = list()
    l_test_logit = list()
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_idx))):
            idx = test_idx[i]
            G.nodes[target_type].data['inp'] = tmp_emb
            G.nodes[target_type].data['inp'][idx] = equal_vec

            logit = model(G, target_type)
            scores = np.take_along_axis(logit, test_indices[i], axis=0).reshape(-1)
            loss = F.mse_loss(scores, test_labels[i].to(device))
            l_test_loss.append(loss.cpu().detach().item())
            l_test_logit.append(logit.cpu().detach().reshape(-1).tolist())

        test_rmse = np.sqrt(sum(l_test_loss) / len(test_idx))
        argsorted_logits_test = np.argsort(np.array(l_test_logit), axis=1)
        test_ndcg = compute_ndcg(test_indices, argsorted_logits_test)
        print(' LR: %.5f Test RMSE %.4f, Test NDCG %.4f' % (
            optimizer.param_groups[0]['lr'],
            test_rmse.item(),
            test_ndcg
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training GNN')
    parser.add_argument('--data', type=str, default='acm')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_epoch', type=int, default=10)
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--n_inp', type=int, default=256)
    parser.add_argument('--clip', type=int, default=1.0)
    parser.add_argument('--max_lr', type=float, default=1e-3)

    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.gpu))

    # acm
    data = scipy.io.loadmat('ACM.mat')
    our_G = dgl.heterograph({
        ('paper', 'written-by', 'author'): data['PvsA'].nonzero(),
        ('author', 'writing', 'paper'): data['PvsA'].transpose().nonzero(),
        ('paper', 'accepted-by', 'conference'): data['PvsC'].nonzero(),
        ('conference', 'accepting', 'paper'): data['PvsC'].transpose().nonzero(),
        ('author', 'joining', 'field'): data['AvsF'].nonzero(),
        ('field', 'joined-by', 'author'): data['AvsF'].transpose().nonzero(),
    })

    start_time = time.time()
    # "pafap"
    pa = data['PvsA']
    af = data['AvsF']
    fa = data['AvsF'].transpose()
    ap = data['PvsA'].transpose()
    pathsim_matrix = pa.dot(af).dot(fa).dot(ap)
    our_labels = calculate_pathsim(pathsim_matrix)
    print("non zero elements count:", np.nonzero(our_labels)[0].shape)
    main(our_G, our_labels)

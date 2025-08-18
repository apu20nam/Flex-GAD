import os
import logging
import torch
from torch_geometric.utils import to_dense_adj, to_networkx
import numpy as np
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import csv
import subprocess
import argparse
import itertools
from pygod.utils import load_data

from auto_encoder import *
from utils import *

from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from pygod.metric import eval_roc_auc
from torch_geometric.data import Data
import wandb



def get_free_gpu():
    gpu_status = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader", shell=True
    )
    free_memory = [int(x) for x in gpu_status.decode("utf-8").strip().split("\n")]
    return free_memory.index(max(free_memory))  


def train(data, y, yc, ys, yj, ysj, lr, epoch, device, encoder, 
          lambda_loss1, lambda_loss2, hidden_dim,
          sample_size=10, loss_step=20, real_loss=False, 
          calculate_contextual=False, calculate_structural=False):

    start_time = time.time()

    # wandb.init(                                        //If want to plot the loss curve in wandb then uncomment this part
    #     project="gnn-loss-logging",
    #     name=f"run-lr{lr}-hdim{hidden_dim}",
    #     config={
    #         "learning_rate": lr,
    #         "epochs": epoch,
    #         "encoder": encoder,
    #         "lambda_loss1": lambda_loss1,
    #         "lambda_loss2": lambda_loss2,
    #         "hidden_dim": hidden_dim
    #     },
    # )

    in_nodes = data.edge_index[0, :]
    out_nodes = data.edge_index[1, :]

    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = [len(neighbor_dict[i]) for i in neighbor_dict]
    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)

    in_dim = data.x.shape[1]
    print(f"attr_dim: {in_dim}")

    GNNModel = GNNStructEncoder(
        in_dim, hidden_dim, hidden_dim, 2, sample_size, device=device,
        neighbor_num_list=neighbor_num_list, GNN_name=encoder,
        lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2
    )
    GNNModel.to(device)

    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params, GNNModel.parameters())

    opt = torch.optim.Adam(
        [{'params': base_params}, {'params': GNNModel.degree_decoder.parameters(), 'lr': 1e-2}],
        lr=lr, weight_decay=0.0003
    )

    min_loss = float('inf')
    best_auc = 0
    best_h_loss_weight, best_feature_loss_weight = None, None

    for i in tqdm(range(1, epoch + 1)):
        loss, loss_per_node, h_loss_per_node, feature_loss_per_node, \
        h_loss, feature_loss, attn_importance = GNNModel(
            data.edge_index, data.x, neighbor_num_list, neighbor_dict, 
            data.adj, data.norm_adj, data.x_new, device
        )

        loss_per_node = loss_per_node.cpu().detach()
        h_loss_per_node = h_loss_per_node.cpu().detach()
        feature_loss_per_node = feature_loss_per_node.cpu().detach()

        h_loss_norm = h_loss_per_node / (torch.max(h_loss_per_node) - torch.min(h_loss_per_node))
        feature_loss_norm = feature_loss_per_node / (torch.max(feature_loss_per_node) - torch.min(feature_loss_per_node))

        h_loss_weight, feature_loss_weight = attn_importance.tolist()

        comb_loss = h_loss_weight * h_loss_norm + feature_loss_weight * feature_loss_norm
        comp_loss = loss_per_node if real_loss else comb_loss

        comp_loss = np.clip(comp_loss, a_min=-1e6, a_max=1e6)
        auc_score = eval_roc_auc(y.numpy(), comp_loss.numpy()) * 100

        if auc_score > best_auc:
            best_auc = auc_score
            best_h_loss_weight = h_loss_weight
            best_feature_loss_weight = feature_loss_weight

        # wandb.log({
        #     "epoch": i,                                       //If want to plot the loss curve in wandb then uncomment this part
        #     "total_loss": loss.item(),
        #     "h_loss": h_loss.item(),
        #     "feature_loss": feature_loss.item(),
        #     "auc_combined": auc_score
        # })

        opt.zero_grad()
        loss.backward()
        opt.step()

    # wandb.finish()                                            //If want to plot the loss curve in wandb then uncomment this part
    total_time = time.time() - start_time
    return best_auc, best_h_loss_weight, best_feature_loss_weight, total_time



def train_real_datasets(dataset_str, epoch_num, lr, encoder, 
                        lambda_loss1, lambda_loss2, sample_size, hidden_dim,
                        real_loss, calculate_contextual, calculate_structural, device):
    
    if dataset_str=="Amazon" or dataset_str =="cora"or dataset_str=="Flickr":


        edge_index, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels = load_mat(dataset_str)

        edge_index = torch.from_numpy(edge_index).long()
        feat = torch.from_numpy(feat).float()

        data = Data(
            x=feat,
            edge_index=edge_index,
            num_nodes=feat.shape[0],
            y=torch.from_numpy(ano_labels)
        )

    else:
        data = load_data(dataset_str)
        edge_index = data.edge_index
        # num_nodes = node_features.shape[0]

    # Normalize features
    node_features = data.x
    node_features_min = node_features.min()
    node_features_max = node_features.max()
    node_features = (node_features - node_features_min) / (node_features_max - node_features_min)
    data.x = node_features

    # Community features
    G = to_networkx(data, to_undirected=True)
    communities = nx_comm.louvain_communities(G)
    new_features = torch.zeros_like(node_features)
    for i, community in enumerate(communities):
        community_nodes = torch.tensor(list(community))
        community_mean = node_features[community_nodes].mean(dim=0)
        normalized_mean = community_mean / community_mean.norm(p=2)
        new_features[community_nodes] = normalized_mean
    data.x_new = new_features

    # Adjacency matrix
    num_nodes = node_features.shape[0]
    adj_matrix = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    data.adj = adj_matrix
    adj_matrix.fill_diagonal_(1)
    degree = adj_matrix.sum(1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    data.norm_adj = degree_inv_sqrt.unsqueeze(1) * adj_matrix * degree_inv_sqrt.unsqueeze(0)

    y = data.y.bool().cpu().detach()
    edge_index = torch.cat([edge_index, torch.arange(num_nodes).repeat(2, 1)], dim=1)
    data.edge_index = edge_index
    data = data.to(device)

    return train(
        data, y, [], [], [], [], lr=lr, epoch=epoch_num, device=device, encoder=encoder,
        lambda_loss1=lambda_loss1, lambda_loss2=lambda_loss2, hidden_dim=hidden_dim,
        sample_size=sample_size, real_loss=real_loss,
        calculate_contextual=calculate_contextual, calculate_structural=calculate_structural
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Anomaly Detection Training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., Amazon, Reddit)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lambda_loss1", type=float, default=0.1)
    parser.add_argument("--lambda_loss2", type=float, default=0.4)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--encoder", type=str, default="SeriesEncoder")
    parser.add_argument("--real_loss", action="store_true")
    parser.add_argument("--calculate_contextual", action="store_true")
    parser.add_argument("--calculate_structural", action="store_true")
    args = parser.parse_args()

    free_gpu = get_free_gpu()
    torch.cuda.set_device(free_gpu)
    device = torch.device(f"cuda:{free_gpu}")
    print(f"Using GPU {free_gpu}: {torch.cuda.get_device_name(free_gpu)}")

    score, best_h_loss_weight, best_feature_loss_weight, total_time = train_real_datasets(
        dataset_str=args.dataset,
        lr=args.lr,
        epoch_num=args.num_epochs,
        lambda_loss1=args.lambda_loss1,
        lambda_loss2=args.lambda_loss2,
        encoder=args.encoder,
        sample_size=args.sample_size,
        hidden_dim=args.dimension,
        real_loss=args.real_loss,
        calculate_contextual=args.calculate_contextual,
        calculate_structural=args.calculate_structural,
        device=device
    )

    print("\n==== Finished ====")
    print(f"Dataset: {args.dataset}")
    print(f"AUC Score: {score:.4f}")
    # print(f" h_loss_weight: {best_h_loss_weight}")
    # print(f" feature_loss_weight: {best_feature_loss_weight}")
    print(f"Total Time: {total_time:.2f} sec")

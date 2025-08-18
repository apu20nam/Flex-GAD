import sys
sys.path.append("..")
from utils import *
import torch
from pygod.utils.utility import check_parameter
from scipy.linalg import sqrtm
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import random
from torch_geometric.data import Data


def _normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min)/(x_max-x_min)
    return x_norm



def gen_joint_structural_outliers(data, m, n, random_state=None):
    """
    We randomly select n nodes from the network which will be the anomalies 
    and for each node we select m nodes from the network. 
    We connect each of n nodes with the m other nodes.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name='m')
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name='n')
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name='m*n')

    if random_state:
        np.random.seed(random_state)


    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    all_nodes = [i for i in range(data.num_nodes)]
    rem_nodes = []
    
    for node in all_nodes:
        if node is not outlier_idx:
            rem_nodes.append(node)
    
    
    
    new_edges = []
    
    # connect all m nodes in each clique
    for i in range(0, n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))
                    

    new_edges = torch.cat(new_edges)


    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier


def KL_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()
    
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)
    
    nn = x1.shape[0]
    h_dim = x1.shape[1]
    
    cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
    cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)
    
    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye
    
    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim  + torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) 
            + (mean_x2 - mean_x1).reshape(1,-1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    KL_loss = KL_loss.to(device)
    return KL_loss

def W2_neighbor_loss(predictions, targets, mask_len):
    
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()
    
    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    
    cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / (nn-1)
    cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / (nn-1)
    

    W2_loss = torch.square(mean_x1-mean_x2).sum() + torch.trace(cov_x1 + cov_x2 
                     + 2 * sqrtm(sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1))))

    return W2_loss




def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot




def load_mat(dataset, train_rate=0.3, val_rate=0.1):
    data = sio.loadmat(f"/home/apuchakraborty/Flex-GAD/dataset/{dataset}.mat")
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    # Convert to sparse matrices
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    # Convert sparse matrices to dense numpy arrays
    feat = feat.toarray()
    
    # Create proper edge_index format
    edges = adj.nonzero()
    edge_index = np.vstack((edges[0], edges[1]))

    # Convert labels
    labels = np.squeeze(np.array(label))
    ano_labels = np.squeeze(np.array(label))
    str_ano_labels = np.squeeze(np.array(data['str_anomaly_label'])) if 'str_anomaly_label' in data else None
    attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label'])) if 'attr_anomaly_label' in data else None

    # Create train/val/test splits
    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    return edge_index, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels
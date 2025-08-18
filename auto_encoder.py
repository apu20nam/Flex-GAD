from torch import  nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
import math
from torch_geometric.nn import GATConv  # Import GATConv instead of GCNConv
import torch

from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv,PNAConv, GraphSAGE

import torch.multiprocessing as mp


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)
                
                if len(h.shape) > 2:
                    h = torch.transpose(h, 0, 1)
                    h = torch.transpose(h, 1, 2)
                    
                h = self.batch_norms[layer](h)
                
                if len(h.shape) > 2:
                    h = torch.transpose(h, 1, 2)
                    h = torch.transpose(h, 0, 1)

                h = F.relu(h)
                # h = F.relu(self.linears[layer](h))
                
            return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=10):
        
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]
    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


# FNN
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        x = F.relu(x)
        return x


class SeriesEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(SeriesEncoder, self).__init__()
        
        self.num_layers = num_layers
        
        # Input layer
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.input_ln = nn.LayerNorm(hidden_channels)  # Add LayerNorm after input
        
        # GCN layers
        self.gcn_convs = nn.ModuleList()
        self.gcn_lns = nn.ModuleList()  # Replace BatchNorm with LayerNorm
        for _ in range(num_layers):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_lns.append(nn.LayerNorm(hidden_channels))
            
        # GIN layers
        self.gin_convs = nn.ModuleList()
        self.gin_lns = nn.ModuleList()  # Replace BatchNorm with LayerNorm
        for _ in range(num_layers):
            gin_nn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),  # Add LayerNorm in GIN MLPs
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.gin_convs.append(GINConv(gin_nn, train_eps=True))
            self.gin_lns.append(nn.LayerNorm(hidden_channels))
            
        # Final linear layerself.reset_parameters()
        self.lin = nn.Linear(hidden_channels, out_channels)
        
        # Residual weights
        self.residual_weight = nn.Parameter(torch.FloatTensor(in_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.residual_weight.size(1))
        self.residual_weight.data.uniform_(-stdv, stdv)

         

    def forward(self, x, adj, edge_index):
        # Input layer with LayerNorm
        h0 = self.input_ln(F.relu(self.input_lin(x)))
        
        # Graph raw residual
        residual = torch.sparse.mm(adj, torch.mm(x, self.residual_weight))
        
        # GCN layers with LayerNorm and residual connections
        x = h0
        for i in range(self.num_layers):
            x_new = self.gcn_convs[i](x, edge_index)
            x_new = F.relu(self.gcn_lns[i](x_new))
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x_new + residual
            x_gcn=x
            
        # GIN layers with LayerNorm and residual connections
        for i in range(self.num_layers):
            x_new = self.gin_convs[i](x, edge_index)
            x_new = F.relu(self.gin_lns[i](x_new))
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x_new + residual
            x_gin=x
            
        # Final linear layer
        x = self.lin(x)
        return x,residual


from torch_geometric.nn import GCNConv
from torch_scatter import scatter


from torch_geometric.nn import GCNConv
from torch_scatter import scatter

from torch_geometric.nn import GCNConv
from torch_scatter import scatter

class GCNConvSum(GCNConv):
    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 add_self_loops=True, normalize=True, bias=True):
        super(GCNConvSum, self).__init__(in_channels, out_channels,
                                         improved=improved, cached=cached,
                                         add_self_loops=add_self_loops,
                                         normalize=normalize, bias=bias)

    def forward(self, x, edge_index):
        if self.normalize:
            edge_index, norm = GCNConv.norm(
                edge_index, x.size(self.node_dim), x.dtype,
                self.improved, self.add_self_loops, self.flow, x.device
            )
        else:
            norm = None

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
    




   


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool  
import math

class GCNEncoder2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, use_sum_agg=True):
        super(GCNEncoder2, self).__init__()
        self.num_layers = num_layers
        self.use_sum_agg = use_sum_agg  

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.gcn_convs = nn.ModuleList()
        self.gcn_lns = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_convs.append(GCNConv(hidden_channels, hidden_channels))
            self.gcn_lns.append(nn.LayerNorm(hidden_channels))    #nn.BatchNorm1d(hidden_channels)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.add_pool_norm = nn.LayerNorm(out_channels)  
        # Residual weights
        self.residual_weight = nn.Parameter(torch.FloatTensor(in_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.residual_weight.size(1))
        self.residual_weight.data.uniform_(-stdv, stdv)

    def forward(self, x,x_org, adj, edge_index, batch=None):  
       
        h0 = F.relu(self.input_lin(x))

        
        residual = torch.sparse.mm(adj, torch.mm(x_org, self.residual_weight))

        
        x = h0
        for i in range(self.num_layers):
            x_new = self.gcn_convs[i](x, edge_index)
            x=global_add_pool(x,batch)
            x_new = F.relu(self.gcn_lns[i](x_new))
            x_new = F.dropout(x_new, p=0.2, training=self.training)
            x = x_new

        
        x = self.lin(x)
        
        

        return x,residual
    

# import os
# import torch
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# class EmbeddingVisualizer:
#     def __init__(self, method='tsne', sample_size=None, save_dir="/home/apuchakraborty/GNN_unified_code/Plot"):
#         """
#         Args:
#             method (str): 'tsne' or 'pca'
#             sample_size (int or None): number of nodes to sample for visualization
#             save_dir (str or None): directory to save plots (if provided)
#         """
#         self.method = method.lower()
#         self.sample_size = sample_size
#         self.save_dir = save_dir

#         if self.method not in ['tsne', 'pca']:
#             raise ValueError("method must be 'tsne' or 'pca'")

#         self.reducer = TSNE(n_components=2, random_state=42, perplexity=5) if self.method == 'tsne' else PCA(n_components=2)

#         if self.save_dir:
#             os.makedirs(self.save_dir, exist_ok=True)

#     def _prepare_embeddings(self, embeddings):
#         """Sample and convert embeddings to numpy if needed."""
#         if self.sample_size is not None and embeddings.size(0) > self.sample_size:
#             idx = torch.randperm(embeddings.size(0))[:self.sample_size]
#             embeddings = embeddings[idx]
#         return embeddings.detach().cpu().numpy()

#     def plot(self, embeddings, title='Embeddings', color=None, filename=None):
#         """
#         Visualize the embeddings and optionally save the figure.

#         Args:
#             embeddings (torch.Tensor): Shape [num_nodes, feature_dim]
#             title (str): Plot title
#             color (array-like or None): Optional coloring
#             filename (str or None): Optional filename to save the plot (without extension)
#         """
#         embeddings_np = self._prepare_embeddings(embeddings)
#         reduced = self.reducer.fit_transform(embeddings_np)

#         plt.figure(figsize=(8, 6))
#         if color is not None:
#             plt.scatter(reduced[:, 0], reduced[:, 1], c=color, cmap='Spectral', s=10, alpha=0.7)
#         else:
#             plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)

#         plt.title(title)
#         plt.xlabel("Component 1")
#         plt.ylabel("Component 2")
#         plt.grid(True)
#         plt.tight_layout()

#         if self.save_dir and filename:
#             save_path = os.path.join(self.save_dir, f"{filename}.png")
#             plt.savefig(save_path)
#             print(f"Saved plot to {save_path}")

#         plt.close()


import os
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap  # UMAP module
from torch_geometric.nn.models import InnerProductDecoder

class EmbeddingVisualizer:
    def __init__(self, method='umap', sample_size=None, save_dir="/home/apuchakraborty/GNN_unified_code/Plot"):
        """
        Args:
            method (str): 'umap' or 'pca'
            sample_size (int or None): number of nodes to sample for visualization
            save_dir (str or None): directory to save plots (if provided)
        """
        self.method = method.lower()
        self.sample_size = sample_size
        self.save_dir = save_dir

        if self.method not in ['umap', 'pca']:
            raise ValueError("method must be 'umap' or 'pca'")

        if self.method == 'umap':
            self.reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        else:
            self.reducer = PCA(n_components=2)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def _prepare_embeddings(self, embeddings):
        """Sample and convert embeddings to numpy if needed."""
        if self.sample_size is not None and embeddings.size(0) > self.sample_size:
            idx = torch.randperm(embeddings.size(0))[:self.sample_size]
            embeddings = embeddings[idx]
        return embeddings.detach().cpu().numpy()

    def plot(self, embeddings, title='Embeddings', color=None, filename=None):
        """
        Visualize the embeddings and optionally save the figure.

        Args:
            embeddings (torch.Tensor): Shape [num_nodes, feature_dim]
            title (str): Plot title
            color (array-like or None): Optional coloring
            filename (str or None): Optional filename to save the plot (without extension)
        """
        embeddings_np = self._prepare_embeddings(embeddings)
        reduced = self.reducer.fit_transform(embeddings_np)

        plt.figure(figsize=(8, 6))
        if color is not None:
            plt.scatter(reduced[:, 0], reduced[:, 1], c=color, cmap='Spectral', s=10, alpha=0.7)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7)

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.tight_layout()

        if self.save_dir and filename:
            save_path = os.path.join(self.save_dir, f"{filename}.png")
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")

        plt.close()


    



class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, h1,h2):
        stacked = torch.stack([h1,h2], dim=0)  # [3, batch_size, embed_dim]
        attn_output, attn_weights = self.multihead_attn(
            stacked, stacked, stacked,
            need_weights=True,
            average_attn_weights=False
        )  
        
        attn_weights_avg = attn_weights.mean(dim=1)
        attn_weights_avg = attn_weights_avg.mean(dim=0)
        attn_received = attn_weights_avg.sum(dim=0)  # [src_len=3]
        attn_importance = attn_received
        h1_new, h2_new = attn_output[0], attn_output[1]

        return h1_new, h2_new, attn_importance





# class GATEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=8):
#         super(GATEncoder, self).__init__()
#         self.num_layers = num_layers
#         self.heads = heads
        
#         # Adjust input linear layer to account for heads
#         self.input_lin = nn.Linear(in_channels, hidden_channels * heads)
        
#         # Initialize GAT layers
#         self.gat_convs = nn.ModuleList()
#         self.gat_bns = nn.ModuleList()
        
#         # First GAT layer after input
#         self.gat_convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
#         self.gat_bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
#         # Middle GAT layers
#         for _ in range(num_layers - 2):
#             self.gat_convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
#             self.gat_bns.append(nn.BatchNorm1d(hidden_channels * heads))
        
#         # Final GAT layer
#         self.gat_convs.append(GATConv(hidden_channels * heads, out_channels // heads, heads=heads, concat=True))
#         self.gat_bns.append(nn.BatchNorm1d(out_channels))
        
#         # Final linear projection
#         self.lin = nn.Linear(out_channels, out_channels)
        
#         # Residual weights
#         self.residual_weight = nn.Parameter(torch.FloatTensor(in_channels, hidden_channels * heads))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.residual_weight.size(1))
#         self.residual_weight.data.uniform_(-stdv, stdv)

#     def forward(self, x, adj, edge_index):
#         # Input projection
#         h0 = F.relu(self.input_lin(x))
        
#         # Graph raw residual
#         residual = torch.sparse.mm(adj, torch.mm(x, self.residual_weight))
        
#         # GAT layers with residual connections
#         x = h0
#         for i in range(self.num_layers):
#             x_new = self.gat_convs[i](x, edge_index)
#             x_new = self.gat_bns[i](x_new)
#             x_new = F.relu(x_new)
#             x_new = F.dropout(x_new, p=0.2, training=self.training)
#             x=x_new
        
#         # Final linear projection
#         x = self.lin(x)
#         return x
    
class AttrEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., act=F.relu, bias=True):
        super(AttrEncoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, inputs):
        x = self.dropout_layer(inputs)
        output = self.linear(x)
        return self.act(output)
    




    # generate ground truth neighbors Hv
def generate_gt_neighbor(neighbor_dict, node_embeddings, neighbor_num_list, in_dim):
    max_neighbor_num = max(neighbor_num_list)
    all_gt_neighbor_embeddings = []
    for i, embedding in enumerate(node_embeddings):
        neighbor_indexes = neighbor_dict[i]
        neighbor_embeddings = []
        for index in neighbor_indexes:
            neighbor_embeddings.append(node_embeddings[index].tolist())
        if len(neighbor_embeddings) < max_neighbor_num:
            for _ in range(max_neighbor_num - len(neighbor_embeddings)):
                neighbor_embeddings.append(torch.zeros(in_dim).tolist())
        all_gt_neighbor_embeddings.append(neighbor_embeddings)
    return all_gt_neighbor_embeddings





# Main Autoencoder structure here
class GNNStructEncoder(nn.Module):
    def __init__(self, in_dim0, in_dim, hidden_dim, layer_num, sample_size, device, neighbor_num_list, 
                 GNN_name="SeriesEncoder", norm_mode="PN-SCS", norm_scale=20, lambda_loss1=0.01, lambda_loss2=0.001):
        
        super(GNNStructEncoder, self).__init__()
        
        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        # self.lambda_loss3 = lambda_loss3
        # GNN Encoder
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(self.linear1)
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(self.linear2)
        elif GNN_name=="SeriesEncoder":
            #self.encoder2 = SeriesEncoder(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=hidden_dim)
            self.encoder1 = GCNEncoder2(in_channels=in_dim, hidden_channels=hidden_dim, out_channels=hidden_dim,use_sum_agg=True)
            self.linear = nn.Linear(3*hidden_dim, hidden_dim)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
            self.graphconv2 = GCNConv(hidden_dim, hidden_dim)
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
            self.graphconv2 = GATConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr='sum')
            # self.graphconv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
            
            # self.graphconv1 = GraphSAGE(hidden_dim, hidden_dim, aggr='mean', num_layers=1)
            
        self.encoder3 = AttrEncoder(input_dim=in_dim, output_dim=hidden_dim, dropout=0.2, act=F.relu)
        self.adjDecoder = InnerProductDecoder()

        self.self_attention = SelfAttentionLayer(hidden_dim, num_heads=1)
        #self.visualizer = EmbeddingVisualizer(method='umap', sample_size=None, save_dir="/home/apuchakraborty/GNN_unified_code/Plot")

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim,
                                                                                     0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(-0.5 / hidden_dim,
                                                                                     0.5 / hidden_dim)).to(device)
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 3)
        self.w1 = nn.Parameter(torch.randn(hidden_dim, hidden_dim * 2))  # (2X, 3X)
        self.w2 = nn.Parameter(torch.randn(hidden_dim, hidden_dim*2))  # (X, 2X)
        self.w3 = nn.Parameter(torch.randn(hidden_dim*2, hidden_dim))  # (2X, X)
        self.w4 = nn.Parameter(torch.randn(hidden_dim*3, hidden_dim*2))  # (3X, 2X)

        torch.nn.init.normal_(self.w1, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.w2, mean=0.0, std=0.01)

        self.m = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            torch.ones(sample_size, hidden_dim))
        
        self.m_batched = torch.distributions.Normal(torch.zeros(sample_size, self.tot_node, hidden_dim),
                                            torch.ones(sample_size, self.tot_node, hidden_dim))

        self.m_h = torch.distributions.Normal(torch.zeros(sample_size, hidden_dim),
                                            50* torch.ones(sample_size, hidden_dim))

        # Before MLP Gaussian Means, and std

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim), torch.ones(hidden_dim))

        self.mlp_mean = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mlp_sigma = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.softplus = nn.Softplus()

        self.mean_agg = SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize = False)
        # self.mean_agg = GraphSAGE(hidden_dim, hidden_dim, aggr='mean', num_layers=1)
        self.std_agg = PNAConv(hidden_dim, hidden_dim, aggregators=["std"],scalers=["identity"], deg=neighbor_num_list)        
        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim)
        
        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size 
        self.init_projection = FNN(in_dim, hidden_dim, hidden_dim, 1)
        

    def forward_encoder(self, x,adj,x_new,edge_index,device):
        h0 = self.mlp0(x)
        h1=self.mlp0(x_new)
        l1,residual=self.encoder1(h1,h0,adj,edge_index)
        # l2,residual = self.encoder2(h0,adj,edge_index)
        l1=l1+residual
        # self.visualizer.plot(x_gcn, title='GCN Output Embeddings', filename='gcn_output')
        # self.visualizer.plot(x_gin, title='GIN Output Embeddings', filename='gin_output')
        
        # print(f"shape_of_l1:{l1.shape}")
        z=self.encoder3(h0)
        # print(f"shape_of_z:{z.shape}")
        # z_a=torch.cat((l1,z),-1)
        # print(f"shape_of_z_a:{z_a.shape}")
        l1,z,attn_importance= self.self_attention(l1,z)

        return h0,h1,l1,z,attn_importance
        
        

    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes, self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)
        
        return sampled_embeddings_list, mark_len_list
    
    def reconstruction_neighbors2(self, l1, h0, edge_index):
        """
        Numerically stable neighbor reconstruction with JSD
        """
        # Small constant for numerical stability
        eps = 1e-8
        
        # Compute means and standard deviations
        mean_neigh = self.mean_agg(h0, edge_index)
        std_neigh = torch.abs(self.std_agg(h0, edge_index)) + eps  # Ensure positive std
        
        # Generate samples with gradient clipping
        self_embedding = l1.unsqueeze(0).repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(self_embedding)
        generated_sigma = torch.clamp(self.mlp_sigma(self_embedding), min=-10, max=10)  # Prevent extreme values
        
        # Sample using reparameterization with gradient clipping
        std_z = torch.clamp(self.m_batched.sample().to(self_embedding.device), min=-5, max=5)
        var = generated_mean + torch.clamp(generated_sigma.exp(), min=eps) * std_z
        nhij = self.layer1_generator(var)
        
        # Compute statistics with numerical safeguards
        generated_mean = torch.mean(nhij, dim=0)
        generated_std = torch.std(nhij, dim=0) + eps
        
        def safe_gaussian_jsd(mu1, sigma1, mu2, sigma2):
            """
            Numerically stable JSD computation in log-space
            """
            # Ensure positive variances
            var1 = torch.clamp(sigma1.pow(2), min=eps)
            var2 = torch.clamp(sigma2.pow(2), min=eps)
            
            # Compute mean distribution parameters
            mu_m = 0.5 * (mu1 + mu2)
            var_m = torch.clamp(0.5 * (var1 + var2), min=eps)
            
            def safe_kl_div(mu_a, var_a, mu_b, var_b):
                """
                Compute KL divergence in a numerically stable way
                """
                # Compute terms separately for better numerical stability
                var_ratio = torch.clamp(var_a / var_b, min=eps)
                mu_diff_sq = (mu_b - mu_a).pow(2)
                var_b_safe = torch.clamp(var_b, min=eps)
                
                # Log terms
                log_var_ratio = torch.log(var_b_safe) - torch.log(var_a.clamp(min=eps))
                
                # Combine terms with careful handling of numerical issues
                kl = 0.5 * (
                    var_ratio + 
                    mu_diff_sq / var_b_safe - 
                    1.0 + 
                    log_var_ratio
                )
                
                # Sum across feature dimension
                return torch.clamp(kl.sum(dim=-1), min=0.0)
            
            # Compute JSD components
            kl1 = safe_kl_div(mu1, var1, mu_m, var_m)
            kl2 = safe_kl_div(mu2, var2, mu_m, var_m)
            
            # Average KL divergences
            jsd = 0.5 * (kl1 + kl2)
            
            # Final numerical safeguard
            return torch.clamp(jsd, min=0.0, max=1e6)
        
        # Compute JSD loss with all safeguards
        jsd_loss = safe_gaussian_jsd(
            mean_neigh,
            std_neigh,
            generated_mean,
            generated_std
        )
        
        # Check for NaN values and replace with small positive value if found
        jsd_loss = torch.where(torch.isnan(jsd_loss), 
                              torch.ones_like(jsd_loss) * eps,
                              jsd_loss)
        
        return torch.mean(jsd_loss), jsd_loss
    
                
        # self_embedding_min = self_embedding.min()
        # self_embedding_max = self_embedding.max()
        # self_embedding = (self_embedding - self_embedding_min)/(self_embedding_max+1e-14)
        # generated_mean = self.mean_generator(self_embedding)
        # generated_mean = _normalize(generated_mean)
        # generated_std = self.std_generator(self_embedding)
        # generated_std = _normalize(generated_std)
        # generated_std = generated_std.abs().squeeze()
        # generated_std = self.softplus(generated_std)
        # generated_std = generated_std.exp().squeeze()
        
        # cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
        # cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)
        
        # generated_cov = generated_cov + torch.rand(generated_cov.shape).to(device)
        # generated_cov = generated_cov * batch_eye
        
        # print(torch.isfinite(det_target_cov).all())
        # print(torch.isfinite(det_generated_cov).all()) 
        
        # KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim  + torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) 
        #     + (mean_x2 - mean_x1).reshape(1,-1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    

    def reconstruction_neighbors(self, FNN_generator, neighbor_indexes, neighbor_dict, from_layer, to_layer, device):
        
        
        local_index_loss = 0
        local_index_loss_per_node = []
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(neighbor_indexes, neighbor_dict, to_layer)
        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick
            
            # print(len(neighbor_embeddings1))
            
            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            nhij = FNN_generator(var)
            
            generated_neighbors = nhij
            sum_neighbor_norm = 0
            
            for indexi, generated_neighbor in enumerate(generated_neighbors):
                sum_neighbor_norm += torch.norm(generated_neighbor) / math.sqrt(self.out_dim)
            generated_neighbors = torch.unsqueeze(generated_neighbors, dim=0).to(device)
            target_neighbors = torch.unsqueeze(torch.FloatTensor(neighbor_embeddings1), dim=0).to(device)
            
            if args.neigh_loss == "KL":
                KL_loss = KL_neighbor_loss(generated_neighbors, target_neighbors, mask_len1)
                local_index_loss += KL_loss
                local_index_loss_per_node.append(KL_loss)
            
            else:
                W2_loss = W2_neighbor_loss(generated_neighbors, target_neighbors, mask_len1)
                local_index_loss += W2_loss
                local_index_loss_per_node.append(W2_loss)
            
            
        local_index_loss_per_node = torch.stack(local_index_loss_per_node)
        return local_index_loss, local_index_loss_per_node
    

    def neighbor_decoder(self,H1,H2,adj, ground_truth_degree_matrix, h0, neighbor_dict, device, h, edge_index):
        
        # Degree decoder below:
        tot_nodes = H1.shape[0]
        #degree_logits = self.degree_decoding(H1)
        #ground_truth_degree_matrix = torch.unsqueeze(ground_truth_degree_matrix, dim=1)
        # degree_loss = self.degree_loss_func(degree_logits, ground_truth_degree_matrix.float())
        # degree_loss_per_node = (degree_logits-ground_truth_degree_matrix).pow(2)
        # _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0
        # layer 1
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(3):
            local_index_loss_sum = 0
            local_index_loss_sum_per_node = []
            indexes = []
            h0_prime = self.feature_decoder(H2)
            feature_losses = self.feature_loss_func(h0, h0_prime)
            feature_losses_per_node = (h0-h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)
            

            
            # local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors(self.layer1_generator, indexes, neighbor_dict, gij, h0, device)
            local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors2(H1,h0,edge_index)
            
            # print(local_index_loss, local_index_loss2)
            # print(local_index_loss_per_node, local_index_loss_per_node2)

            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)
            
        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)
        
        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node,dim=0)
        
        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))
        # print(f"edge_index shape: {edge_index.shape}")


        # adj_reconstructed = self.adjDecoder.forward_all(H2)
        # adj_loss_per_node = (adj_reconstructed - adj).pow(2).mean(1)
        # print(f"adj_loss_per_node shape: {adj_loss_per_node.shape}")
        # adj_loss=adj_loss_per_node.mean()
        # print(f"adj_loss : {adj_loss}")

        
        

        
        # adj_loss_per_node = adj_loss_per_node.reshape(tot_nodes,1)      
        h_loss_per_node = h_loss_per_node.reshape(tot_nodes,1)
        #degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes,1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes,1)
        # print(f"lambda_loss1: {self.lambda_loss1}, lambda_loss2: {self.lambda_loss2}")
        
        loss = self.lambda_loss1 * h_loss  + self.lambda_loss2 * feature_loss
        loss_per_node = self.lambda_loss1 * h_loss_per_node  + self.lambda_loss2 * feature_loss_per_node
        

        return loss, loss_per_node, h_loss_per_node,  feature_loss_per_node,h_loss,feature_loss






    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(self, edge_index, x, ground_truth_degree_matrix, neighbor_dict,norm_adj,adj,x_new,device):
        h0,h,l1,z,attn_importance= self.forward_encoder(x,norm_adj,x_new, edge_index,device)
        h1,h2= l1,z
        #print(f'h1: {h1.shape}, h2: {h2.shape}, h3: {h3.shape}')
        h_concat = torch.cat((h1,h2), dim=-1)
        z_n = F.linear(h_concat, self.w1, bias=None)
        # # z_n = F.linear(h_reduced, self.w2, bias=None)
        z_n_1 = F.linear(z_n, self.w3, bias=None)
        # #z_n_2= F.linear(z_n_1, self.w4, bias=None)
        H1,H2=torch.split(z_n_1, self.out_dim, dim=-1)
        # print(f'H1: {H1.shape}, H2: {H2.shape}')

        
    #     # Concatenate h1, h2, h3 along the last dimension
    #     h_concat = torch.cat((h1, h2, h3), dim=-1)  # Shape: [N, 3X]
    #     #h_concat = self.layer_norm(h_concat) 
        
    #     # First MLP (3X → 2X) with GELU activation
    #     h_reduced = torch.nn.functional.gelu(F.linear(h_concat, self.w1, bias=None))  # Shape: [N, 2X]
        
    #     # Second MLP (2X → 1X)
    #     z_n = F.linear(h_reduced, self.w2, bias=None)
    #     # h1,h2,h3 = l1, l2,z
    # #     print(f'h1:{h1.shape} ,h2:{h2.shape}')

    # #     z_n = (
    # #     h1 @ self.w1.T + 
    # #     h2 @ self.w2.T + 
    # #     h3 @ self.w3.T 
    # # )
    #     # z_n = (
    #     #     h1 @ self.w1.T + 
    #     #     h2 @ self.w2.T + 
    #     #     (h1*h2) @ self.w3.T 
    #     #  )
        
    #     # z_n=self.linear(z_n)
    #     # print(f'shape of z_n:{z_n.shape}')
    #     #reconstructed_attr = self.attrDecoder(z_n,z_a)
    #     #print(f"reconstructed_attr shape: {reconstructed_attr.shape}")
    #     z_n=z_n
    #     print(f'the shape of z_n:{z_n.shape}')
    #     print(f"shape_of_x:{x.shape}")
    #     #attr_recos_loss = F.mse_loss(reconstructed_attr, x)


        
        loss, loss_per_node, h_loss_per_node,  feature_loss_per_node,h_loss,feature_loss = self.neighbor_decoder(H1,H2, adj,ground_truth_degree_matrix, h0, neighbor_dict, device, x, edge_index)
        #loss=loss+attr_recos_loss*self.lambda_loss2
        return loss, loss_per_node, h_loss_per_node,  feature_loss_per_node,h_loss,feature_loss,attn_importance
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.data
from torch_geometric.nn import GATConv, VGAE


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels)
        self.conv_mu = GATConv(2 * out_channels, out_channels)
        self.conv_logstd = GATConv(2 * out_channels, out_channels)


    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def main():
    model_name = 'VGAE'
    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    disease_disease_file = 'data/MimMiner/MimMiner.txt'
    gene_gene_file = 'data/HumanNetV2/HumanNet_V2.txt'
    prediction_files = [f'data/prediction/{model_name}/prediction-3-fold-1.txt',
                        f'data/prediction/{model_name}/prediction-3-fold-2.txt',
                        f'data/prediction/{model_name}/prediction-3-fold-3.txt']

    for counter in [3]:
        g_nx = nx.Graph()
        with open(disease_gene_files[counter], 'r') as f:
            for line in f:
                node1, node2, tag = line.strip().split('\t')
                if tag == 'train':
                    g_nx.add_node(node1)
                    g_nx.add_node(node2)
                    g_nx.add_edge(node1, node2, weight=1)
        with open(gene_gene_file, 'r') as f:
            for line in f:
                node1, node2 = line.strip().split('\t')
                g_nx.add_node(node1)
                g_nx.add_node(node2)
                g_nx.add_edge(node1, node2, weight=1)
        with open(disease_disease_file, 'r') as f:
            for line in f:
                node1, node2, weight = line.strip().split('\t')
                g_nx.add_node(node1)
                g_nx.add_node(node2)
                g_nx.add_edge(node1, node2, weight=1)
        print('read data success')

        name_id = dict(zip(g_nx.nodes(), range(g_nx.number_of_nodes())))
        g_nx = nx.relabel_nodes(g_nx, name_id)

        # transform from networkx to pyg data
        g_nx = g_nx.to_directed() if not nx.is_directed(g_nx) else g_nx
        edge_index = torch.tensor(list(g_nx.edges)).t().contiguous()
        data = {}
        data['edge_index'] = edge_index.view(2, -1)
        data = torch_geometric.data.Data.from_dict(data)
        data.num_nodes = g_nx.number_of_nodes()
        data.x = torch.from_numpy(np.eye(data.num_nodes)).float()
        data.train_mask = data.val_mask = data.test_mask = data.y = None
        print(f'Graph information:\nNode:{data.num_nodes}\nEdge:{data.num_edges}\nFeature:{data.num_node_features}')

        channels = 128
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VGAE(Encoder(data.num_node_features, channels)).to(dev)
        x, train_pos_edge_index = data.x.to(dev), data.edge_index.to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(4000):
            model.train()
            optimizer.zero_grad()
            z = model.encode(x, train_pos_edge_index)
            loss = model.recon_loss(z, train_pos_edge_index) + (1 / data.num_nodes) * model.kl_loss()
            loss.backward()
            optimizer.step()
            nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'{nowTime}\tepoch:{epoch}\tloss:{loss}')


        z = model.encode(x, train_pos_edge_index)
        pred = model.decoder.forward_all(z).cpu().detach().numpy().tolist()

        id_name = {}
        diseases = set()
        genes = set()
        for key in name_id:
            id_name[name_id[key]] = key
            if key.startswith('g_'):
                genes.add(key)
            elif key.startswith('d_'):
                diseases.add(key)

        test_diseases = set()
        with open(disease_gene_files[counter], 'r') as f:
            for line in f:
                disease, gene, tag = line.strip().split('\t')
                if tag == 'test':
                    test_diseases.add(disease)

        with open(prediction_files[counter], 'w') as f:
            for disease in test_diseases:
                sims = {}
                if disease not in diseases:
                    for gene in genes:
                        sims[gene] = 0
                else:
                    for gene in genes:
                        sim = pred[name_id[disease]][name_id[gene]]
                        sims[gene] = sim
                sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
                c = 0
                for gene, sim in sorted_sims:
                    f.write(disease + '\t' + gene + '\t' + str(sim) + '\n')
                    c += 1
                    if c >= 150:
                        break


if __name__ == '__main__':
    main()

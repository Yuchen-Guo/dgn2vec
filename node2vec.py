import datetime
import itertools
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed

from prediction import prediction


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.alias_nodes = {}
        self.alias_edges = {}
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr]['weight'] for nbr in sorted(self.G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self.alias_setup(normalized_probs)
        if self.is_directed:
            for edge in self.G.edges():
                self.alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in self.G.edges():
                self.alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                self.alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])


    def node2vec_walk(self, walk_length, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next_node = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next_node)
            else:
                break

        return walk


    @staticmethod
    def partition_num(num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            result = [num // workers] * workers
            for i in range(num % workers):
                result[i] += 1
            return result


    def simulate_walks(self, num_walks, walk_length):
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\tsimulate walk starts...')
        G = self.G
        nodes = list(G.nodes())
        workers = 6
        results = Parallel(n_jobs=workers)(delayed(self._simulate_walks)(nodes=nodes, num_walks=num, walk_length=walk_length) for num in self.partition_num(num_walks, workers))
        return list(itertools.chain(*results))


    def _simulate_walks(self, nodes, num_walks, walk_length):
        walks = []
        for walk_iter in range(num_walks):
            print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\t{walk_iter + 1}/{num_walks}')
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks


    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)


    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


    @staticmethod
    def alias_setup(probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q


    @staticmethod
    def alias_draw(J, q):
        K = len(J)

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]


def main():
    model_name = 'node2vec'
    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    disease_disease_file = 'data/MimMiner/MimMiner.txt'
    gene_gene_file = 'data/HumanNetV2/HumanNet_V2.txt'
    embedding_files = [f'data/prediction/{model_name}/Graph-3-fold-1.emb',
                       f'data/prediction/{model_name}/Graph-3-fold-2.emb',
                       f'data/prediction/{model_name}/Graph-3-fold-3.emb']
    prediction_files = [f'data/prediction/{model_name}/prediction-3-fold-1.txt',
                        f'data/prediction/{model_name}/prediction-3-fold-2.txt',
                        f'data/prediction/{model_name}/prediction-3-fold-3.txt']

    p = 1.2
    q = 2
    num_walks = 100
    walk_length = 40
    dimensions = 128

    for counter in range(1):
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
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\tread data success')

        G = Graph(g_nx, False, p, q)
        walks = G.simulate_walks(num_walks, walk_length)

        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(sentences=walks, size=dimensions, window=10, min_count=0, sg=1, workers=8, iter=1)
        model.wv.save_word2vec_format(embedding_files[counter])
        prediction(embedding_files[counter], disease_gene_files[counter], prediction_files[counter])


if __name__ == "__main__":
    main()

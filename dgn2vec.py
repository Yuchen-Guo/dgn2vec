import itertools
import random

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed

from prediction import prediction


class Graph:

    def __init__(self, G):
        self.G = G


    def partition_num(self, num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            result = [num // workers] * workers
            for i in range(num % workers):
                result[i] += 1
            return result


    def simulate_walks(self, num_walks, walk_length, p):
        G = self.G
        nodes = list(G.nodes())

        # return self._simulate_walks(nodes=nodes, num_walks=num_walks, walk_length=walk_length, p=p)

        # 并行
        workers = 8
        results = Parallel(n_jobs=workers)(delayed(self._simulate_walks)(nodes=nodes, num_walks=num, walk_length=walk_length, p=p) for num in self.partition_num(num_walks, workers))
        return list(itertools.chain(*results))


    def _simulate_walks(self, nodes, num_walks, walk_length, p):
        walks = []
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.metapath_walk(walk_length=walk_length, start_node=node, p=p))
        return walks


    def metapath_walk(self, walk_length, start_node, p):
        walk = [start_node]
        current_node = start_node
        G = self.G
        metapath = self.genMetapath(current_type=G.nodes[start_node]['label'], walk_length=walk_length, p=p)
        
        for i in range(walk_length - 1):
            neighbors = sorted(G.neighbors(current_node))
            neighbors_meta = []
            for neighbor in neighbors:
                if G.nodes[neighbor]['label'] == metapath[i + 1]:
                    neighbors_meta.append(neighbor)

            if len(neighbors) != 0:
                if len(neighbors_meta) == 0:
                    probs = []
                    for neighbor in neighbors:
                        probs.append(G[current_node][neighbor]['weight'])
                    prob_sum = sum(probs)
                    normalized_probs = [prob / prob_sum for prob in probs]
                    current_node = np.random.choice(neighbors, 1, p=normalized_probs)[0]
                else:
                    probs = []
                    for neighbor in neighbors_meta:
                        probs.append(G[current_node][neighbor]['weight'])
                    prob_sum = sum(probs)
                    normalized_probs = [prob / prob_sum for prob in probs]
                    current_node = np.random.choice(neighbors_meta, 1, p=normalized_probs)[0]
                walk.append(current_node)
            else:
                break
        return walk


    def genMetapath(self, current_type, walk_length, p):
        metapath = [current_type]
        # p = self.p
        for i in np.random.choice([0, 1], walk_length-1, p=[1-p, p]):
            if i == 1:
                metapath.append(current_type)
            else:
                current_type = self.differentType(current_type)
                metapath.append(current_type)
        return metapath


    def differentType(self, current_type):
        if current_type == 'd':
            return 'g'
        elif current_type == 'g':
            return 'd'


def main():
    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    disease_disease_file = 'data/MimMiner/MimMiner.txt'
    gene_gene_file = 'data/HumanNetV2/HumanNet_V2.txt'

    embedding_files = ['data/prediction/dgn2vec/Graph-3-fold-1.emb',
                       'data/prediction/dgn2vec/Graph-3-fold-2.emb',
                       'data/prediction/dgn2vec/Graph-3-fold-3.emb']
    prediction_files = ['data/prediction/dgn2vec/prediction-3-fold-1.txt',
                        'data/prediction/dgn2vec/prediction-3-fold-2.txt',
                        'data/prediction/dgn2vec/prediction-3-fold-3.txt']

    walk_length = 40
    num_walks = 100
    p = 0.3  # 转移到同类型节点的概率

    for i in range(3):

        g_nx = nx.Graph()
        with open(disease_gene_files[i], 'r') as f:
            for line in f:
                node1, node2, tag = line.strip().split('\t')
                if tag == 'train':
                    g_nx.add_node(node1, label=node1[0])
                    g_nx.add_node(node2, label=node2[0])
                    g_nx.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=1)
        with open(gene_gene_file, 'r') as f:
            for line in f:
                node1, node2 = line.strip().split('\t')
                g_nx.add_node(node1, label=node1[0])
                g_nx.add_node(node2, label=node2[0])
                g_nx.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=1)
        with open(disease_disease_file, 'r') as f:
            for line in f:
                node1, node2, weight = line.strip().split('\t')
                g_nx.add_node(node1, label=node1[0])
                g_nx.add_node(node2, label=node2[0])
                g_nx.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=float(weight))

        G = Graph(g_nx)
        walks = G.simulate_walks(num_walks=num_walks, walk_length=walk_length, p=p)
        model = Word2Vec(sentences=walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)
        model.wv.save_word2vec_format(embedding_files[i])
        prediction(embedding_files[i], disease_gene_files[i], prediction_files[i])


if __name__ == '__main__':
    main()

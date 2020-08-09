import  networkx as nx
import numpy as np

gene_num = 0
disease_num = 0

def L1_norm(matrix):
    matrix /= matrix.sum(axis=1)[:,None]
    np.nan_to_num(matrix, copy=False)


def main():
    global gene_num, disease_num
    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    disease_disease_file = 'data/MimMiner/MimMiner.txt'
    gene_gene_file = 'data/HumanNetV2/HumanNet_V2.txt'
    
    prediction_files = ['data/prediction/RWRH/prediction-3-fold-1.txt',
                        'data/prediction/RWRH/prediction-3-fold-2.txt',
                        'data/prediction/RWRH/prediction-3-fold-3.txt']

    transfer_probability  = 0.7
    restart_probability = 0.7
    importance = 0.5
    L1_threshold = 1e-10

    for count in range(3):
        G = nx.Graph()
        gene_set = set()
        disease_set = set()

        with open(disease_gene_files[count], 'r') as f:
            for line in f:
                node1, node2, tag = line.strip().split('\t')
                if tag == 'train':
                    G.add_node(node1, label=node1[0])
                    G.add_node(node2, label=node2[0])
                    G.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=1)
                    for node in [node1, node2]:
                        if node[0] == 'g':
                            gene_set.add(node)
                        elif node[0] == 'd':
                            disease_set.add(node)
        with open(gene_gene_file, 'r') as f:
            for line in f:
                node1, node2 = line.strip().split('\t')
                G.add_node(node1, label=node1[0])
                G.add_node(node2, label=node2[0])
                G.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=1)
                for node in [node1, node2]:
                    if node[0] == 'g':
                        gene_set.add(node)
                    elif node[0] == 'd':
                        disease_set.add(node)
        with open(disease_disease_file, 'r') as f:
            for line in f:
                node1, node2, weight = line.strip().split('\t')
                G.add_node(node1, label=node1[0])
                G.add_node(node2, label=node2[0])
                G.add_edge(node1, node2, label=node1[0] + '_' + node2[0], weight=float(weight))
                for node in [node1, node2]:
                    if node[0] == 'g':
                        gene_set.add(node)
                    elif node[0] == 'd':
                        disease_set.add(node)

        gene_list = list(gene_set)
        disease_list = list(disease_set)
        gene_id = {}
        disease_id = {}

        gene_count = 0
        disease_count = 0
        for gene in gene_list:
            gene_id[gene] = gene_count
            gene_count += 1
        for disease in disease_list:
            disease_id[disease] = disease_count
            disease_count += 1

        gene_num = len(gene_list)
        disease_num = len(disease_list)

        print('gene_num:', gene_num)
        print('disease_num:', disease_num)

        M = nx.convert_matrix.to_numpy_array(G, nodelist=gene_list+disease_list)
        gene_gene = M[:gene_num, :gene_num]
        gene_disease = M[:gene_num, gene_num:gene_num+disease_num]
        disease_gene = M[gene_num:gene_num+disease_num, :gene_num]
        disease_disease = M[gene_num:gene_num + disease_num, gene_num:gene_num + disease_num]

        L1_norm(gene_gene)
        gene_gene *= 1 - transfer_probability
        L1_norm(gene_disease)
        gene_disease *= transfer_probability
        L1_norm(disease_gene)
        disease_gene *= transfer_probability
        L1_norm(disease_disease)
        disease_disease *= 1 - transfer_probability
        L1_norm(M)

        test_diseases = set()
        with open(disease_gene_files[count], 'r') as f:
            for line in f:
                disease, gene, tag = line.strip().split('\t')
                if tag == 'test':
                    test_diseases.add(disease)
        print('total test diseases:', len(test_diseases))

        with open(prediction_files[count], 'w') as f:

            test_disease_num = 0

            for test_disease in test_diseases:

                ##
                test_disease_num += 1
                print('test disease num:', test_disease_num, end='\t')
                ##

                if test_disease not in disease_list:
                    print('test_disease not in disease_list')
                    continue
                else:
                    print('')
                    test_disease_id = disease_id[test_disease]
                    u0 = disease_gene[test_disease_id][:,None]
                    v0 = disease_disease[test_disease_id][:,None]
                    u0 = u0 / np.sum(u0, axis=0)[0]
                    u0 = np.nan_to_num(u0)
                    v0 = v0 / np.sum(v0, axis=0)[0]
                    v0 = np.nan_to_num(v0)
                    p0 = np.concatenate(((1-importance) * u0, importance * v0), axis=0)
                    p = p0
                    L1 = 100
                    while L1 > L1_threshold:
                        temp = p
                        p = (1 - restart_probability) * np.dot(M.T, p) + restart_probability * p0
                        L1 = np.linalg.norm(p, ord=1) - np.linalg.norm(temp, ord=1)

                    sims = {}
                    for i in range(gene_num):
                        sims[gene_list[i]] = p[i][0]
                    sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)

                    counter = 0
                    for gene, sim in sorted_sims:
                        if sim == 0:
                            break
                        f.write(test_disease + '\t' + gene + '\t' + str(sim) + '\n')
                        counter += 1
                        if counter >= 150:
                            break


if __name__ == '__main__':
    main()

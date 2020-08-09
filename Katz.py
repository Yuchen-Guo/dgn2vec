import networkx as nx


def main():
    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    disease_disease_file = 'data/MimMiner/MimMiner.txt'
    gene_gene_file = 'data/HumanNetV2/HumanNet_V2.txt'
    
    prediction_files = ['data/prediction/Katz/prediction-3-fold-1.txt',
                        'data/prediction/Katz/prediction-3-fold-2.txt',
                        'data/prediction/Katz/prediction-3-fold-3.txt']


    constant = 0.001


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

        M = nx.convert_matrix.to_numpy_array(G, nodelist=gene_list + disease_list)
        gene_gene = M[:gene_num, :gene_num]
        gene_disease = M[:gene_num, gene_num:gene_num + disease_num]
        disease_gene = M[gene_num:gene_num + disease_num, :gene_num]
        disease_disease = M[gene_num:gene_num + disease_num, gene_num:gene_num + disease_num]

        ggd = gene_gene.dot(gene_disease)
        gdd = gene_disease.dot(disease_disease)

        score = constant * gene_disease + constant * constant * (ggd + gdd) \
                + constant * constant * constant * (gene_disease.dot(disease_gene).dot(gene_disease) + gene_gene.dot(ggd) 
                                                    + gene_gene.dot(gdd) + gdd.dot(disease_disease))


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
                    sims = {}
                    for i in range(gene_num):
                        sims[gene_list[i]] = score[:,test_disease_id][i]
                    sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)

                    counter = 0
                    for gene, sim in sorted_sims:
                        f.write(test_disease + '\t' + gene + '\t' + str(sim) + '\n')
                        counter += 1
                        if counter >= 150:
                            break


if __name__ == '__main__':
    main()
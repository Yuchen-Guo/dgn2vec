import numpy as np
import math


class DaDa:
    def __init__(self):
        self.gene_NI = {}   # key: gene name, value: gene id
        self.gene_IN = {}  # key: gene id, value: gene name
        self.dis_sim_set = {}    # key: dis, value: a dic, key: similar dis value sim score
        self.ppi_net = np.zeros([1, 1])
        self.ppi_net_norm = np.zeros([1, 1])
        self.net_size = 0
        self.train_data = {}   # train data of dis-gene
        self.test_data = {}     # test data of dis-gene
        self.sig = 'CENT'
        self.hyb = 'SEED'
        self.dis_sim_thld = 0.4     # similarity threshold
        self.c = 0.3    # crosstalk restart probability. 0.3 is the optimal value
        self.dis_top_thld = 100   # top disease threshold
        self.test_dis_seed = {}
        self.ppi_net_deg = []
        self.ppi_net_avg_deg = 0


    def run(self, ppi_file, dis_sim_file, disease_gene_file, param, out_file):
        self.load_ppi(ppi_file)
        self.load_dis_sim_file(dis_sim_file)
        self.load_disease_gene_file(disease_gene_file)
        self.set_param(param)
        self.column_normalize()
        self.get_net_deg()
        self.pred_gene(out_file)


    def pred_gene(self, out_file):
        with open(out_file, 'w') as f:
            dis_counter = 0
            self.test_dis_seed = self.get_seed_gene()
            test_dis_dic = self.test_data['dis_dic']

            train_gene_set = self.train_data['gene']
            train_dis_dic = self.train_data['dis_dic']

            test_dis_num = len(test_dis_dic)
            for test_dis, test_genes in test_dis_dic.items():
                dis_counter += 1
                print(dis_counter, '/', test_dis_num, ':', test_dis, end='\t')

                if test_dis not in  train_dis_dic:
                    print('test_dis not in  train_dis_dic')
                    continue

                if test_dis not in self.test_dis_seed:
                    print('no seed genes')
                    continue
                else:
                    train_genes = train_dis_dic[test_dis]
                    predicted_genes = self.pred_g(test_dis)
                    if predicted_genes is None:
                        print('predicted_genes is None')
                        continue
                    counter = 0
                    for score, gid in predicted_genes:
                        gene_name = self.gene_IN[gid]
                        score_std = (500-score)/500
                        if gene_name in train_gene_set and gene_name not in train_genes:
                            f.write('\t'.join([test_dis, self.gene_IN[gid], str(score_std)]) + '\n')
                            counter += 1
                            if counter >= 150:
                                break
                    print('')


    def pred_g(self, test_dis):
        seed_genes = self.test_dis_seed[test_dis]
        if len(seed_genes) == 0:
            return

        seed_gid = []
        for gene, score in seed_genes.items():
            if gene in self.gene_NI:
                gid = self.gene_NI.get(gene)
                seed_gid.append([gid, score])
        if len(seed_gid) == 0:
            return
        seed_avg_deg = self.get_avg_seed_deg(seed_gid)

        c_vector, _ = self.seed_crosstalkers_nodisease(self.ppi_net, self.c, seed_gid)
        c_vector_ss = [c_vector[i, 0] for i in range(len(c_vector))]

        if self.sig == 'CENT':
            # print('statistical adjustment method: based on network centrality')
            p_vector, _ = self.seed_crosstalkers_nodisease(self.ppi_net, 0, seed_gid)
            combined_vector = c_vector / p_vector
            sig_vector = [combined_vector[i, 0] for i in range(len(combined_vector)) ]
        else:
            print('Error. No such option!')
            return

        if self.hyb == 'SEED':
            dada = self.hybrid_seeddegree_sort(c_vector_ss, sig_vector, 'descend', seed_avg_deg)
            # print('uniform prioritization method: based on average seed degree')
        elif self.hyb == 'OPT':
            m, dada = self.hybrid_best_sort(c_vector_ss, sig_vector, 'descend', self.net_size)
            # print('uniform prioritization method: best ranking for each gene (optimistic)')
        elif self.hyb == 'CAND':
            m, dada = self.hybrid_candidatedegree_sort(c_vector_ss, sig_vector, 'descend')
            # print('uniform prioritization method: based on average candidate degree')
        else:
            print('Error. No such option!')
            return

        return dada


    def get_seed_gene(self):
        print('get_seed_gene...')
        test_dis_dic = self.test_data['dis_dic']
        train_dis_dic = self.train_data['dis_dic']

        test_dis_seed = {}   # key: test dis, value: seed genes, a list
        for test_dis in test_dis_dic:
            if test_dis not in self.dis_sim_set:
                continue
            similar_diseases = self.dis_sim_set.get(test_dis)
            sorted_diseases = sorted(similar_diseases.items(), key=lambda x: x[1], reverse=True)
            for dis, sim in sorted_diseases:
                test_dis_seed.setdefault(test_dis, {})
                genes = train_dis_dic.get(dis)
                if len(test_dis_seed[test_dis]) == self.dis_top_thld: break
                if genes is not None:
                    for g in genes:
                        if len(test_dis_seed[test_dis]) == self.dis_top_thld: break
                        if g not in test_dis_seed[test_dis]:
                            test_dis_seed[test_dis][g] = sim

        return test_dis_seed


    def hybrid_seeddegree_sort(self, c_vector, sig_vector, mode, seed_avg_deg):
        merged_vect = []

        c_vector_temp = self.sort2(c_vector, mode)
        sig_vector_temp = self.sort2(sig_vector, mode)

        for i in range(self.net_size):
            c_number = c_vector[i]
            m1 = self.get_m(c_vector_temp, c_number)
            sig_number = sig_vector[i]
            m2 = self.get_m(sig_vector_temp, sig_number)
            if seed_avg_deg < self.ppi_net_avg_deg:
                merged_vect.append(m2)
            else:
                merged_vect.append(m1)

        m, I = self.my_sort(merged_vect, self.net_size-1, 'ascend')
        return I


    def get_m(self, vect, number):
        pos_list = vect[number]
        min_value = min(pos_list)
        max_value = max(pos_list)
        if min_value == max_value:
            m = min_value
            return m
        else:
            m = math.floor((min_value + max_value)/2)
            return m


    def sort2(self, vect, mode):
        vect_size = len(vect)
        value_pos = {}
        vect_temp = [[vect[i], i] for i in range(vect_size)]
        if mode == 'descend':
            vect_temp.sort(reverse=True)
        elif mode == 'ascend':
            vect_temp.sort(reverse=False)
        I = vect_temp
        for i in range(vect_size):
            pos = i
            value = vect_temp[i][0]
            value_pos.setdefault(value, [])
            value_pos[value].append(pos)
        return value_pos


    def my_sort(self, vect, index, mode):
        number = vect[index]
        s = len(vect)
        vect_temp = [[vect[i], i] for i in range(len(vect))]
        if mode == 'descend':
            vect_temp.sort(reverse=True)
        elif mode == 'ascend':
            vect_temp.sort(reverse=False)
        sorted_vect = [a for a, _ in vect_temp]
        I = vect_temp

        first_pos = sorted_vect.index(number)  # 第一出现number的位置
        for i in range(first_pos, s):
            if sorted_vect[i] != number:
                l = i - 1
                m = math.floor((first_pos + l) / 2)
                return m, I
        l = s
        m = math.floor((first_pos + l) / 2)
        return m, I


    def set_param(self, param):
        if param.get('SIG') is not None:
            self.sig = param.get('SIG')
        if param.get('HYB') is not None:
            self.hyb = param.get('HYB')
        if param.get('dis_sim_thld') is not None:
            self.dis_sim_thld = param.get('dis_sim_thld')
        if param.get('dis_top_thld') is not None:
            self.dis_top_thld = param.get('dis_top_thld')


    def load_dis_sim_file(self, dis_sim_file):
        with open(dis_sim_file, 'r') as fr:
            for line in fr:
                d1, d2, sim = line.strip().split('\t')
                sim = float(sim)
                self.dis_sim_set.setdefault(d1, {d2: sim})
                self.dis_sim_set.setdefault(d2, {d1: sim})
                self.dis_sim_set[d1][d2] = sim
                self.dis_sim_set[d2][d1] = sim


    def load_ppi(self, ppi_file):
        print('read ppi data ...')
        edge = []
        gene_num = 0
        with open(ppi_file, 'r') as fr:
            for line in fr:
                p1, p2 = line.strip().split('\t')
                if p1 not in self.gene_NI:
                    self.gene_NI[p1] = gene_num
                    self.gene_IN[gene_num] = p1
                    gene_num += 1
                if p2 not in self.gene_NI:
                    self.gene_NI[p2] = gene_num
                    self.gene_IN[gene_num] = p2
                    gene_num += 1
                p1_id = self.gene_NI[p1]
                p2_id = self.gene_NI[p2]
                edge.append([p1_id, p2_id])
        gene_num = len(self.gene_NI)
        self.net_size = gene_num
        self.ppi_net = np.zeros([gene_num, gene_num])
        for p1_id, p2_id in edge:
            self.ppi_net[p1_id, p2_id] = 1
            self.ppi_net[p2_id, p1_id] = 1


    def column_normalize(self):
        print('column_normalize...')
        for i in range(self.net_size):
            deg = sum(self.ppi_net[:, i])
            if deg == 0:
                break
            else:
                self.ppi_net[:, i] = self.ppi_net[:, i] / np.linalg.norm(self.ppi_net[:, i], 1)
        return self.ppi_net


    def load_disease_gene_file(self, disease_gene_file):
        train_edge_set = set()
        test_edge_set = set()
        train_gene_set = set()
        test_gene_set = set()
        train_dis_dic = {}  # key: test dis_name, value: all the test genes of the dis
        test_dis_dic = {}  # key: train dis_name, value: all the train genes of the dis
        with open(disease_gene_file, 'r') as f:
            for line in f:
                disease, gene, tag = line.strip().split('\t')
                if tag == 'train':
                    if disease not in train_dis_dic:
                        train_dis_dic[disease] = {gene}
                    else:
                        train_dis_dic[disease].add(gene)
                    train_edge_set.add(disease + '\t' + gene)
                    train_gene_set.add(gene)
                if tag == 'test':
                    if disease not in test_dis_dic:
                        test_dis_dic[disease] = {gene}
                    else:
                        test_dis_dic[disease].add(gene)
                    test_edge_set.add(disease + '\t' + gene)
                    test_gene_set.add(gene)

        self.train_data['gene'] = train_gene_set
        self.train_data['edge'] = train_edge_set
        self.train_data['dis_dic'] = train_dis_dic
        self.test_data['gene'] = test_gene_set
        self.test_data['edge'] = test_edge_set
        self.test_data['dis_dic'] = test_dis_dic


    def get_avg_seed_deg(self, seed_gid):
        deg_sum = 0
        for gid, _ in seed_gid:
            deg_sum += sum(self.ppi_net[:, gid])
        deg_avg = deg_sum/len(seed_gid)
        return deg_avg


    def get_net_deg(self):
        for i in range(self.net_size):
            self.ppi_net_deg.append(sum(self.ppi_net[:, i]))
        self.ppi_net_avg_deg = sum(self.ppi_net_deg)/len(self.ppi_net_deg)


    def seed_crosstalkers_nodisease(self, net, c, seed_gid):
        threshold = 1e-10
        maxit = 100
        residue = 1
        iter = 1
        prox_vector = np.zeros([self.net_size, 1])
        for gid, score in seed_gid:
            prox_vector[gid, 0] = score
        prox_vector = prox_vector/np.linalg.norm(prox_vector, 1)
        restart_vector = prox_vector
        while residue > threshold and iter < maxit:
            old_prox_vector = prox_vector
            prox_vector = (1 - c) * np.dot(net, prox_vector) + c * restart_vector
            residue = np.linalg. norm(prox_vector - old_prox_vector, 1)
            iter = iter + 1
        return prox_vector, residue


    def hybrid_best_sort(self, c_vector_subset, sig_vector, mode, candidate_size):
        merged_vect = []
        for i in range(candidate_size):
            m1, _ = self.my_sort(c_vector_subset, i, mode)
            m2, _ = self.my_sort(sig_vector, i, mode)
            merged_vect = merged_vect + min(m1, m2)
        m, I = self.my_sort(merged_vect, candidate_size, 'ascend')
        return m, I


    def hybrid_candidatedegree_sort(self, c_vector_subset, sig_vector, mode):
        merged_vect = []
        for i in range(self.net_size):
            m1, _ = self.my_sort(c_vector_subset, i, mode)
            m2, _ = self.my_sort(sig_vector, i, mode)
            d = self.ppi_net_deg[i]
            if d < self.ppi_net_avg_deg:
                merged_vect = merged_vect + m2
            else:
                merged_vect = merged_vect + m1
        m, I = self.my_sort(merged_vect, self.net_size-1, 'ascend')
        return m, I


def main():
    ppi_file = 'data/HumanNetV2/HumanNet_V2.txt'
    dis_sim_file = 'data/MimMiner/MimMiner.txt'

    disease_gene_files = ['data/OMIM/3-fold-1.txt',
                          'data/OMIM/3-fold-2.txt',
                          'data/OMIM/3-fold-3.txt']
    prediction_files = ['data/prediction/DADA/prediction-3-fold-1.txt',
                        'data/prediction/DADA/prediction-3-fold-2.txt',
                        'data/prediction/DADA/prediction-3-fold-3.txt']

    for i in range(3):
        param = {'SIG': 'CENT', 'HYB': 'SEED', 'dis_sim_thld': 0.4, 'dis_top_thld': 100}
        dada = DaDa()
        dada.run(ppi_file, dis_sim_file, disease_gene_files[i], param, prediction_files[i])


if __name__ == '__main__':
    main()

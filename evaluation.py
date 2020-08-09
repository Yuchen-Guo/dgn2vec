import re


disease_set = set()
gene_set = set()


def load_train_test_file(train_test_file):
    test = {}   # key: disease; value: the gene set
    train = {}  # key: disease; value: the gene set
    with open(train_test_file, 'r') as f:
        for line in f:
            disease, gene, tag = line.strip().split('\t')

            disease_set.add(disease)
            gene_set.add(gene)

            if tag == 'test':
                if disease not in test:
                    test[disease]  = {gene}
                else:
                    test[disease].add(gene)
            if tag == 'train':
                if disease not in train:
                    train[disease] = {gene}
                else:
                    train[disease].add(gene)
    return train, test


def load_prediction_file(prediction_file, train_set, k):
    prediction = {} # key: disease; value: the gene list
    counter = {}
    with open(prediction_file, 'r') as f:
        for line in f:
            disease, gene, score = line.strip().split('\t')
            if disease not in prediction:
                prediction[disease] = []
                counter[disease] = k
            if counter[disease] > 0:
                if disease in train_set and gene in train_set[disease]:
                    continue
                prediction[disease].append(gene)
                counter[disease] -= 1
    return prediction


def evaluation_topk(prediction_set, test_set, k):
    counter = 0
    TP = 0
    FP = 0
    FN = 0
    for disease in test_set:
        if disease not in prediction_set or len(prediction_set[disease]) == 0:
            counter += 1
            # continue
            prediction_set[disease] = set() # 是否考虑未在训练集中出现过的疾病

        if len(prediction_set[disease]) < k:
            prediction_topk = set(prediction_set[disease])
        else:
            prediction_topk = set(prediction_set[disease][:k])

        TP += len(test_set[disease] & prediction_topk)
        FN += len(test_set[disease] - prediction_topk)
        FP += len(prediction_topk - test_set[disease])

    precision = TP / (k * len(test_set))
    recall = TP / (TP + FN)

    return precision, recall


def main():
    prediction_files = ['data/prediction/dgn2vec/prediction-3-fold-1.txt',
                        'data/prediction/dgn2vec/prediction-3-fold-2.txt',
                        'data/prediction/dgn2vec/prediction-3-fold-3.txt']
    train_test_files = ['data/OMIM/3-fold-1.txt',
                        'data/OMIM/3-fold-2.txt',
                        'data/OMIM/3-fold-3.txt']
    evaluation_files = ['data/Evaluation/dgn2vec/3-fold-1.txt',
                        'data/Evaluation/dgn2vec/3-fold-2.txt',
                        'data/Evaluation/dgn2vec/3-fold-3.txt']
    evaluation_mean_file = 'data/Evaluation/dgn2vec/3-fold-mean.txt'

    for i in range(3):
        train, test = load_train_test_file(train_test_files[i])
        prediction = load_prediction_file(prediction_files[i], train, float('inf'))
        with open(evaluation_files[i], 'w') as f:
            for j in range(1, 101, 1):
                precision, recall = evaluation_topk(prediction, test, j)
                print('top %d: precision:%f, recall:%f' % (j, precision, recall))
                f.write('top %d: precision:%f, recall:%f\n' % (j, precision, recall))


    # calculate 3 fold mean
    with open(evaluation_files[0], 'r') as f1:
        with open(evaluation_files[0], 'r') as f2:
            with open(evaluation_files[2], 'r') as f3:
                with open(evaluation_mean_file, 'w') as fw:
                    for i in range(100):
                        line1 = f1.readline()
                        line2 = f2.readline()
                        line3 = f3.readline()
                        temp1 = re.split('[,:\n]', line1)
                        precision1 = float(temp1[2])
                        recall1 = float(temp1[4])
                        temp2 = re.split('[,:\n]', line2)
                        precision2 = float(temp2[2])
                        recall2 = float(temp2[4])
                        temp3 = re.split('[,:\n]', line3)
                        precision3 = float(temp3[2])
                        recall3 = float(temp3[4])
                        precision = (precision1 + precision2 + precision3) / 3
                        recall = (recall1 + recall2 + recall3) / 3
                        fw.write('top %d: precision:%f, recall:%f\n' % (i + 1, precision, recall))


if __name__ == '__main__':
    main()

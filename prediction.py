def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA*normB)**0.5)


def prediction(embedding_file, disease_gene_file, prediction_file):
    diseases = {}
    genes = {}
    test_diseases = set()

    with open(embedding_file, 'r') as f:
        next(f)
        for line in f:
            temp, vec = line.strip().split(' ', 1)
            if temp.startswith('g_'):
                arr = vec.strip().split(' ')
                arr1 = [float(i) for i in arr]
                genes[temp] = arr1
            if temp.startswith('d_'):
                arr = vec.strip().split(' ')
                arr1 = [float(i) for i in arr]
                diseases[temp] = arr1

    with open(disease_gene_file, 'r') as f:
        for line in f:
            disease, gene, tag = line.strip().split('\t')
            if tag == 'test':
                test_diseases.add(disease)

    with open(prediction_file, 'w') as f:
        for disease in test_diseases:
            sims = {}
            if disease not in diseases:
                for gene in genes:
                    sims[gene] = 0
                # continue
            else:
                for gene in genes:
                    sim = cos(diseases[disease], genes[gene])
                    sims[gene] = sim

            sorted_sims = sorted(sims.items(), key=lambda item: item[1], reverse=True)
            counter = 0
            for gene, sim in sorted_sims:
                f.write(disease + '\t' + gene + '\t' + str(sim) + '\n')
                counter += 1
                if counter >= 150:
                    break

import argparse
import numpy as np
import codecs
import pickle
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description='TextLevelGNN-DGL data packaging project')

    # experiment setting
    parser.add_argument('--dataset', type=str, default='ohsumed', choices=['r8', 'r52', 'ohsumed'], help='used dataset ')
    parser.add_argument('--n_degree', required=False, type=int, default=4, help='in/out neighbor node number')
    parser.add_argument('--pretrained', type=bool, default=True, help='use pretrained GloVe embeddings')
    parser.add_argument('--pmi_graph', type=bool, default=False,  help='ablation: use predefined pmi graph')
    parser.add_argument('--d_pretrained', type=int, default=300, help='pretrained embedding dimension')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # path settings
    parser.add_argument('--path_data', type=str, default='./data/', help='path of the data corpus')

    args = parser.parse_args()
    np.random.seed(args.seed)

    if args.dataset not in ['r8', 'r52', 'ohsumed']:
        raise ValueError('Data {data} not supported, currently supports "r8", "r52" and "ohsumed".')

    # read files
    print('\n[info] Dataset:', args.dataset)
    time_start = time.time()

    label2idx = read_label(args.path_data + args.dataset + '/label.txt')
    word2idx = read_vocab(args.path_data + args.dataset + '/vocab-5.txt')
    args.n_class = len(label2idx)
    args.n_word = len(word2idx)
    print('\tTotal classes:', args.n_class)
    print('\tTotal words:', args.n_word)

    embeds = get_embedding(args, word2idx)

    tr_data, tr_gt = read_corpus(args.path_data + args.dataset + '/train-stemmed.txt', label2idx, word2idx)
    print('\n\tTotal training samples:', len(tr_data))

    val_data, val_gt = read_corpus(args.path_data + args.dataset + '/valid-stemmed.txt', label2idx, word2idx)
    print('\tTotal validation samples:', len(val_data))

    te_data, te_gt = read_corpus(args.path_data + args.dataset + '/test-stemmed.txt', label2idx, word2idx)
    print('\tTotal testing samples:', len(te_data))

    edges_weights_pmi, edges_mappings_pmi, edges_mappings, co_occur_matrix = make_graph(args, tr_data)
    print('\tTotal edges:', len(edges_mappings))

    # save processed data
    mappings_pmi = {
        'label2idx': label2idx,
        'word2idx': word2idx,
        'tr_data': tr_data,
        'tr_gt': tr_gt,
        'val_data': val_data,
        'val_gt': val_gt,
        'te_data': te_data,
        'te_gt': te_gt,
        'embeds': embeds,
        'edges_weights': edges_weights_pmi,
        'edges_mappings': edges_mappings_pmi,
        'co_occur_matrix': co_occur_matrix,
        'args': args
    }

    mappings = {
        'label2idx': label2idx,
        'word2idx': word2idx,
        'tr_data': tr_data,
        'tr_gt': tr_gt,
        'val_data': val_data,
        'val_gt': val_gt,
        'te_data': te_data,
        'te_gt': te_gt,
        'embeds': embeds,
        'edges_weights': None,
        'edges_mappings': edges_mappings,
        'co_occur_matrix': co_occur_matrix,
        'args': args
    }

    with open(args.path_data + args.dataset + '_pmi.pkl', 'wb') as f:
        pickle.dump(mappings_pmi, f)
    with zipfile.ZipFile(args.path_data + args.dataset + '_pmi.zip', 'w') as zf:
        zf.write(args.path_data + args.dataset + '_pmi.pkl', compress_type=zipfile.ZIP_DEFLATED)

    with open(args.path_data + args.dataset + '_co_occur.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    with zipfile.ZipFile(args.path_data + args.dataset + '_co_occur.zip', 'w') as zf:
        zf.write(args.path_data + args.dataset + '_co_occur.pkl', compress_type=zipfile.ZIP_DEFLATED)

    print('\n[info] Time consumed: {:.2f}s'.format(time.time() - time_start))


def read_label(path):
    """ Extract and encode labels. """
    with open(path) as f:
        labels = f.read().split('\n')

    return {label: i for i, label in enumerate(labels)}


def read_vocab(path):
    """ Extract words from vocab and encode. """
    with open(path) as f:
        words = f.read().split('\n')
    word2idx = {word: i + 1 for i, word in enumerate(words)}
    word2idx['<pad>'] = 0

    return word2idx


def read_corpus(path, label2idx, word2idx):
    """ Encode both corpus and labels. """
    with open(path) as f:
        content = [line.split('\t') for line in f.read().split('\n')]

    data = [[encode_word(word, word2idx) for word in x[1].split()] for x in content]
    gt = [label2idx[x[0]] for x in content]
    return data, gt


def encode_word(word, word2idx):
    """ Encode word considering unknown word. """
    try:
        idx = word2idx[word]
    except KeyError:
        idx = word2idx['UNK']
    return idx


def get_embedding(args, word2idx):
    """ Find words in pretrained GloVe embeddings. """

    path = args.path_data + 'glove.6B.' + str(args.d_pretrained) + 'd.txt'
    embeds_word = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word2idx), args.d_pretrained))  # initializing
    emb_counts = 0

    if args.pretrained:
        for i, line in enumerate(codecs.open(path, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == (args.d_pretrained + 1) and s[0] in word2idx:
                embeds_word[word2idx[s[0]]] = np.array([float(i) for i in s[1:]])
                emb_counts += 1
    else:
        embeds_word = None
        emb_counts = 'disabled pretrained'

    print('\tPretrained GloVe found:', emb_counts)
    return embeds_word


def make_graph(args, data):
    word_count = np.zeros(args.n_word, dtype=int)
    co_occur_matrix = np.zeros((args.n_word, args.n_word), dtype=int)
    co_occur_matrix_pmi = np.zeros((args.n_word, args.n_word), dtype=int)

    # count co-occurrence
    for sentence in data:
        len_sent = len(sentence)
        for i, word in enumerate(sentence):
            word_count[word] += 1
            for j in range(max(0, i - args.n_degree), min(len_sent, i + args.n_degree)):
                co_occur_matrix[word, sentence[j]] += 1
                if i != j:
                    co_occur_matrix_pmi[word, sentence[j]] += 1
    total_count = np.sum(word_count)

    # calculate pmi matrix
    pmi_matrix = np.zeros((args.n_word, args.n_word), dtype=float)
    for i in range(args.n_word):
        for j in range(args.n_word):
            if co_occur_matrix_pmi[i, j] == 0 or i == 0 or j == 0:
                pmi_matrix[i, j] = -1
            else:
                pmi_matrix[i, j] = np.log(co_occur_matrix_pmi[i, j] * (total_count / (word_count[i] * word_count[j])))
    pmi_matrix = np.maximum(pmi_matrix, 0)  # turn negative number to 0

    # create edge mappings
    edges_weights_pmi = [0.0]
    count_pmi = 1
    edges_mappings_pmi = np.full((args.n_word, args.n_word), 0, dtype=int)
    count = 1
    edges_mappings = np.full((args.n_word, args.n_word), 0, dtype=int)
    for i in range(args.n_word):
        for j in range(args.n_word):
            if pmi_matrix[i, j] > 0:
                edges_weights_pmi.append(pmi_matrix[i, j])
                edges_mappings_pmi[i, j] = count_pmi
                count_pmi += 1

            if co_occur_matrix[i, j] > 0:
                edges_mappings[i, j] = count
                count += 1

    edges_weights_pmi = np.array(edges_weights_pmi).reshape(-1, 1)
    args.n_edge, args.n_edge_pmi = count, count_pmi

    return edges_weights_pmi, edges_mappings_pmi, edges_mappings, co_occur_matrix


if __name__ == '__main__':
    main()

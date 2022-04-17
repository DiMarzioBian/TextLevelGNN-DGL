import os
import pickle
import zipfile
import numpy as np
import torch
import dgl

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, args, data, gt):
        self.max_len_text = args.max_len_text
        self.n_degree = args.n_degree

        self.data = data
        self.gt = gt
        self.length = len(self.gt)

    def get_graph(self):
        return

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.data[idx][:self.max_len_text]
        local_vocab = np.unique(data)
        map_reindex = {idx_old: idx_new for idx_new, idx_old in enumerate(local_vocab)}

        # create edge
        edge_in, edge_out = [], []
        for i, word in enumerate(data):
            for j in range(max(0, i - self.n_degree), min(len(data), i + self.n_degree)):
                edge_in.append(word)
                edge_out.append(data[j])

            edge_in.append(word)
            edge_out.append(word)

        edges = np.transpose([edge_in, edge_out])
        edges = np.array(list(set([tuple(e) for e in edges])))

        edges_reindex = np.array([[map_reindex[edge[0]], map_reindex[edge[1]]] for edge in edges])
        edges_reindex = torch.tensor(edges_reindex)
        g = dgl.graph((edges_reindex[:, 0], edges_reindex[:, 1]), num_nodes=len(local_vocab))

        return g, torch.tensor(local_vocab), edges, self.gt[idx]


def collate_fn(insts, edges_mappings, co_occur_matrix):
    """ Batch process """
    graph_batch, nodes_batch, edges_batch, y_batch = list(zip(*insts))
    graph_batch = dgl.batch(graph_batch)
    nodes_batch = torch.cat(nodes_batch, 0)
    y_batch = torch.LongTensor(y_batch)

    # map edges
    edges_batch = torch.LongTensor([edges_mappings[edge[0], edge[1]] if co_occur_matrix[edge[0], edge[1]] > 0
                                    else edges_mappings[0, 0] for edges in edges_batch for edge in edges])

    return graph_batch, nodes_batch, edges_batch, y_batch


def get_dataloader(args):
    """ Get dataloader, word2idx and pretrained embeddings """

    # read files and extract data
    type_graph = '_pmi' if args.pmi_graph else '_co_occur'
    if not os.path.exists(args.path_data + args.dataset + type_graph + '.pkl'):
        if os.path.exists(args.path_data + args.dataset + type_graph + '.zip'):
            print('\n[info] Found zipped dataset "{data}", start unzipping.'.format(data=args.dataset))
            with zipfile.ZipFile(args.path_data + args.dataset + type_graph + '.zip', 'r') as zip_ref:
                zip_ref.extractall()
        else:
            raise FileNotFoundError('\n[warning] Dataset "{data}" not found.'.format(data=args.dataset))

    with open(args.path_data + args.dataset + type_graph + '.pkl', 'rb') as f:
        mappings = pickle.load(f)

    args_prepare = mappings['args']
    tr_data, tr_gt = mappings['tr_data'], mappings['tr_gt']
    val_data, val_gt = mappings['val_data'], mappings['val_gt']
    te_data, te_gt = mappings['te_data'], mappings['te_gt']
    embeds = torch.FloatTensor(mappings['embeds'])
    edges_mappings = mappings['edges_mappings']
    edges_weights = torch.FloatTensor(mappings['edges_weights']) if args.pmi_graph else None
    co_occur_matrix = torch.LongTensor(mappings['co_occur_matrix'])

    # arguments updates
    if args_prepare.d_pretrained != args.d_model:
        raise ValueError('Experiment settings do not match data preprocess settings: d_pretrained. '
                         'Please re-run prepare.py with correct settings.')
    if args_prepare.n_degree != args.n_degree and args.pmi_graph:
        raise ValueError('PMI graph formed by {n_degree_prepare} neighbors, but experiment takes {n_degree} neighbors.'
                         .format(n_degree_prepare=args_prepare.n_degree, n_degree=args.n_degree))

    args.n_class = args_prepare.n_class
    args.n_word = args_prepare.n_word
    args.n_edge = args_prepare.n_edge_pmi if args.pmi_graph else args_prepare.n_edge

    # edge occurrence thresholding
    co_occur_matrix = torch.where(co_occur_matrix < args.edge_occur_threshold, 0, co_occur_matrix)

    # create dataloader
    tr_loader = DataLoader(TextDataset(args, tr_data, tr_gt), batch_size=args.batch_size, num_workers=args.num_worker,
                           shuffle=True, collate_fn=lambda x: collate_fn(x, edges_mappings, co_occur_matrix))

    val_loader = DataLoader(TextDataset(args, val_data, val_gt), batch_size=args.batch_size, num_workers=args.num_worker,
                            shuffle=True, collate_fn=lambda x: collate_fn(x, edges_mappings, co_occur_matrix))

    te_loader = DataLoader(TextDataset(args, te_data, te_gt), batch_size=args.batch_size, num_workers=args.num_worker,
                           shuffle=True, collate_fn=lambda x: collate_fn(x, edges_mappings, co_occur_matrix))

    return tr_loader, val_loader, te_loader, embeds, edges_mappings, edges_weights

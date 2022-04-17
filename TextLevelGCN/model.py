import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_reduce(node: object) -> object:
    w = node.mailbox['w']

    new_hidden = torch.mul(w, node.mailbox['m'])

    new_hidden, _ = torch.max(new_hidden, 1)

    node_eta = torch.sigmoid(node.data['eta'])
    # node_eta = F.leaky_relu(node.data['eta'])

    # new_hidden = node_eta * node.data['h'] + (1 - node_eta) * new_hidden
    # print(new_hidden.shape)

    return {'h': new_hidden}


class TextLevelGCN(nn.Module):
    def __init__(self, args, embeds, edges_mappings, pmi=None):
        super(TextLevelGCN, self).__init__()

        self.mean_reduction = args.mean_reduction
        self.device = args.device
        self.d_model = args.d_model
        self.n_degree = args.n_degree

        self.embeddings_node = nn.Embedding.from_pretrained(embeds, padding_idx=0)
        if args.pmi_graph:
            self.embeddings_edge = nn.Embedding.from_pretrained(pmi, freeze=True)
        else:
            self.embeddings_edge = nn.Embedding.from_pretrained(torch.ones(args.n_edge, 1), freeze=False)

        self.edges_mappings = edges_mappings
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.n_class, bias=True)

    def forward(self, graph_batch, nodes_batch, edges_batch):
        # load nodes and edges data
        graph_batch.ndata['h'] = self.embeddings_node(nodes_batch)
        graph_batch.edata['w'] = self.embeddings_edge(edges_batch)

        if not self.mean_reduction:
            graph_batch.update_all(
                message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
                reduce_func=dgl.function.max('weighted_message', 'h')
            )
        else:
            graph_batch.update_all(
                message_func=dgl.function.src_mul_edge('h', 'w', 'weighted_message'),
                reduce_func=dgl.function.mean('weighted_message', 'h')
            )

        outputs = dgl.sum_nodes(graph_batch, feat='h')

        outputs = F.relu(self.dropout(outputs))
        outputs = self.fc(outputs)

        return outputs

from tqdm import tqdm
import torch
import torch.nn.functional as F


def train(args, model, data, optimizer):
    model.train()
    loss_total = 0.
    n_sample = 0
    correct_pred_total = 0

    for batch in tqdm(data, desc='  - training', leave=False):
        graph_batch, nodes_batch, edges_batch, y_batch = map(lambda x: x.to(args.device), batch)

        optimizer.zero_grad()
        scores_batch = model(graph_batch, nodes_batch, edges_batch)
        loss_batch = F.cross_entropy(scores_batch, y_batch)
        loss_batch.backward()
        optimizer.step()

        # calculate loss
        loss_total += loss_batch * scores_batch.shape[0]
        n_sample += scores_batch.shape[0]
        correct_pred_total += (scores_batch.max(dim=-1)[1] == y_batch).sum()

    loss_mean = loss_total / n_sample
    acc = correct_pred_total / n_sample

    return loss_mean, acc


def evaluate(args, model, data):
    model.train()
    loss_total = 0.
    n_sample = 0
    correct_pred_total = 0

    with torch.no_grad():
        for batch in tqdm(data, desc='  - evaluating', leave=False):
            graph_batch, nodes_batch, edges_batch, y_batch = map(lambda x: x.to(args.device), batch)

            scores_batch = model(graph_batch, nodes_batch, edges_batch)
            loss_batch = F.cross_entropy(scores_batch, y_batch)

            # calculate loss
            loss_total += loss_batch * scores_batch.shape[0]
            n_sample += scores_batch.shape[0]
            correct_pred_total += (scores_batch.max(dim=-1)[1] == y_batch).sum()

    loss_mean = loss_total / n_sample
    acc = correct_pred_total / n_sample

    return loss_mean, acc

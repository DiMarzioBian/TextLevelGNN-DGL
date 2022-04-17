import os
import time
import argparse
import copy
import numpy as np
import random
import torch
from TextLevelGNN.model import TextLevelGCN
from dataloader import get_dataloader
from epoch import train, evaluate


def main():
    parser = argparse.ArgumentParser(description='TextLevelGNN project')

    # experiment setting
    parser.add_argument('--dataset', type=str, default='r8', choices=['r8', 'r52', 'ohsumed'], help='used dataset')
    parser.add_argument('--n_degree', required=False, type=int, default=4, help='in/out neighbor node number')
    parser.add_argument('--mean_reduction', type=bool, default=False, help='ablation: mean reduction: default max')
    parser.add_argument('--pmi_graph', type=bool, default=True,  help='ablation: use predefined pmi graph')
    parser.add_argument('--pretrained', type=bool, default=True, help='ablation: use pretrained GloVe')
    parser.add_argument('--edge_occur_threshold', type=int, default=2, help='ablation: public edge min. occurrence')

    # hyperparameters
    parser.add_argument('--d_model', type=int, default=300, help='node representation dimensions including embedding')
    parser.add_argument('--max_len_text', type=int, default=100, help='maximum length of text')
    parser.add_argument('--dropout', required=False, type=float, default=0.5, help='dropout rate')
    parser.add_argument('--device', type=str, default='cuda:0',  help='device for computing')

    # training settings
    parser.add_argument('--num_worker', type=int, default=0, help='number of dataloader worker')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--lr_step', type=int, default=10, help='number of epoch for each lr downgrade')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='strength of lr downgrade')
    parser.add_argument('--es_patience_max', type=int, default=10, help='max early stopped patience')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # path settings
    parser.add_argument('--path_data', type=str, default='./data/', help='path of the data corpus')
    parser.add_argument('--path_log', type=str, default='./result/logs/', help='path of the training logs')
    parser.add_argument('--path_model', type=str, default='./result/models/', help='path of the trained model')
    parser.add_argument('--save_model', type=bool, default=False, help='save model for further use')

    args = parser.parse_args()

    if args.dataset not in ['r8', 'r52', 'ohsumed']:
        raise ValueError('Data {data} not supported, currently supports "r8", "r52" and "ohsumed".'
                         .format(data=args.dataset))
    for path in [args.path_log, args.path_model]:
        if not os.path.exists(path):
            os.makedirs(path)

    args.device = torch.device(args.device)
    args.path_log += 'log' + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.txt'
    args.path_model_params = args.path_model + 'model_params' + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.pt'
    args.path_model += 'model_cuda' + str(args.device)[-1] + time.strftime('_%b_%d_%H_%M', time.localtime()) + '.pt'
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # prepare data and model
    print('\n[info] Project starts.')
    tr_loader, val_loader, te_loader, embeds, edges_mappings, edges_weights = get_dataloader(args)

    model = TextLevelGCN(args, embeds, edges_mappings=edges_mappings, pmi=edges_weights
                         ).to(args.device)

    optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # Start modeling
    print('\n[info] | Dataset: {Dataset} | n_degree: {n_degree} | mean_reduction: {mean_reduction} | '
          'pmi_graph: {pmi_graph} | pretrained: {pretrained} | edge_occur_threshold: {edge_occur_threshold} |'
          .format(Dataset=args.dataset, n_degree=args.n_degree, mean_reduction=args.mean_reduction,
                  pmi_graph=args.pmi_graph, pretrained=args.pretrained, edge_occur_threshold=args.edge_occur_threshold))
    loss_best = 1e5
    acc_best = 0
    epoch_best = 0
    es_patience = 0

    for epoch in range(1, args.epochs + 1):
        print('\n[Epoch {epoch}]'.format(epoch=epoch))

        # training phase
        t_start = time.time()
        loss_train, acc_train = train(args, model, tr_loader, optimizer)
        scheduler.step()
        print(' \t| Train | loss {:5.4f} | acc {:5.4f} | {:5.2f} s |'
              .format(loss_train, acc_train, time.time() - t_start))

        # validating phase
        loss_val, acc_val = evaluate(args, model, val_loader)

        # early stopping condition
        if acc_val > acc_best or (acc_val == acc_best and loss_val < loss_best):
            es_patience = 0
            state_best = copy.deepcopy(model.state_dict())
            loss_best = loss_val
            acc_best = acc_val
            epoch_best = epoch
        else:
            es_patience += 1
            if es_patience >= args.es_patience_max:
                print('\n[Warning] Early stopping model')
                print('\t| Best | epoch {:d} | loss {:5.4f} | acc {:5.4f} |'
                      .format(epoch_best, loss_best, acc_best))
                break

        # logging
        print('\t| Valid | loss {:5.4f} | acc {:5.4f} | es_patience {:.0f}/{:.0f} |'
              .format(loss_val, acc_val, es_patience, args.es_patience_max))

    # testing phase
    print('\n[Testing]')
    model.load_state_dict(state_best)
    if args.save_model:
        with open(args.path_model_params, 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(args.path_model, 'wb') as f:
            torch.save(model, f)

    loss_test, acc_test = evaluate(args, model, te_loader)

    print('\n\t| Test | loss {:5.4f} | acc {:5.4f} |'
          .format(loss_test, acc_test))
    print('\n[info] | Dataset: {Dataset} | n_degree: {n_degree} | mean_reduction: {mean_reduction} | '
          'pmi_graph: {pmi_graph} | pretrained: {pretrained} | edge_occur_threshold: {edge_occur_threshold} |\n'
          .format(Dataset=args.dataset, n_degree=args.n_degree, mean_reduction=args.mean_reduction,
                  pmi_graph=args.pmi_graph, pretrained=args.pretrained, edge_occur_threshold=args.edge_occur_threshold))


if __name__ == '__main__':
    main()
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/22 2:16 AM
# @Author: Zechen Li
# @File  : main.py

import argparse
from utils.misc import *
import time
import torch.nn as nn

from utils.train_helpers import *
from utils.test_helpers import test

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./RTE/')
parser.add_argument('--aug_dataroot', default='./augment_data/RTE')
parser.add_argument('--task_name', default='RTE')
parser.add_argument('--model', default='bert-large-uncased')
########################################################################
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--workers', default=8, type=int)
########################################################################
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--weight_decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

args = parser.parse_args()
my_makedir(args.outf)

net, head, ssh = build_model(args)
trloader = prepare_train_data(args)
net_teloader, ssh_teloader = prepare_test_data(args)

parameters = list(net.parameters()) + list(head.parameters())
optimizer = torch.optim.SGD(parameters,
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss(reduction='none').cuda()


def train(trloader, epoch):
    net.train()
    ssh.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(trloader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))

    end = time.time()
    for i, dl in enumerate(trloader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        net_input_ids, net_input_masks, net_segment_ids, net_label_ids = \
            dl[0].cuda(), dl[1].cuda(), dl[2].cuda(), dl[3].cuda()

        outputs_cls = net(input_ids=net_input_ids.long(),
                          attention_mask=net_input_masks.long(),
                          token_type_ids=net_segment_ids.long(),)
        outputs_cls = nn.functional.normalize(outputs_cls[0], dim=1)
        net_label_ids = net_label_ids.long()
        loss_cls = criterion(outputs_cls, net_label_ids)
        loss = loss_cls.mean()
        losses.update(loss.item(), len(net_label_ids))

        _, predicted = outputs_cls.max(1)
        acc1 = predicted.eq(net_label_ids).sum().item() / len(net_label_ids)
        top1.update(acc1, len(net_label_ids))

        ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids = \
            dl[4].cuda(), dl[5].cuda(), dl[6].cuda(), dl[7].cuda()
        outputs_ssh = ssh(input_ids=ssh_input_ids.long(),
                          attention_mask=ssh_input_masks.long(),
                          token_type_ids=ssh_segment_ids.long(),)
        outputs_ssh = nn.functional.normalize(outputs_ssh[0], dim=1)
        ssh_label_ids = ssh_label_ids.long()
        loss_ssh = criterion(outputs_ssh, ssh_label_ids)
        loss += loss_ssh.mean()

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.print(i)


all_err_cls = []
all_err_ssh = []

if args.resume is not None:
    print('Resuming from checkpoint..')
    ckpt = torch.load('%s/ckpt.pth' % (args.resume))
    net.load_state_dict(ckpt['net'])
    head.load_state_dict(ckpt['head'])
    optimizer.load_state_dict(ckpt['optimizer'])
    loss = torch.load('%s/loss.pth' % (args.resume))
    all_err_cls, all_err_ssh = loss

for epoch in range(args.start_epoch, args.epochs+1):
    adjust_learning_rate(optimizer, epoch, args)
    train(trloader, epoch)
    err_cls = test(net_teloader, net)
    err_ssh = test(ssh_teloader, ssh)

    all_err_cls.append(err_cls)
    all_err_ssh.append(err_ssh)
    torch.save((all_err_cls, all_err_ssh), args.outf + '/loss.pth')
    plot_epochs(all_err_cls, all_err_ssh, args.outf + '/loss.pdf')

    state = {'args': args, 'err_cls': err_cls, 'err_ssh': err_ssh,
             'optimizer': optimizer.state_dict(), 'net': net.state_dict(), 'ssh': ssh.state_dict()}
    torch.save(state, args.outf + '/ckpt.pth')

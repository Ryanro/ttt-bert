#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/7/25 7:16 PM
#@Author: Zechen Li
#@File  : test_helpers.py
import time
import torch
import torch.nn as nn
from utils.misc import *


def test(teloader, model, verbose=False, print_freq=10):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(teloader), batch_time, top1, prefix='Test: ')

    one_hot = []
    losses = []
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    end = time.time()

    for i, (input_ids, input_masks, segment_ids, label_ids) in enumerate(teloader):
        with torch.no_grad():
            input_ids, input_masks, segment_ids, label_ids = \
                input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
            outputs = model(input_ids=input_ids.long(),
                            attention_mask=input_masks.long(),
                            token_type_ids=segment_ids.long(),)
            outputs = nn.functional.normalize(outputs[0], dim=1)
            label_ids = label_ids.long()
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, label_ids).cpu())
            one_hot.append(predicted.eq(label_ids).cpu())

        acc1 = one_hot[-1].sum().item() / len(label_ids)
        top1.update(acc1, len(label_ids))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)
    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    if verbose:
        one_hot = torch.cat(one_hot).numpy()
        losses = torch.cat(losses).numpy()
        return 1-top1.avg, one_hot, losses
    else:
        return 1-top1.avg
#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2020/7/25 7:16 PM
#@Author: Zechen Li
#@File  : test_helpers.py
import time
from utils.misc import *
from utils.train_helpers import *


def prepare_test_data(args):
    print('Preparing net evaluating data...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    net_task = get_task(args.task_name, args.dataroot)

    net_examples = net_task.get_dev_examples()

    net_label_list = net_task.get_labels()
    net_label_map = {label: i for i, label in enumerate(net_label_list)}

    net_input_ids = []
    net_input_masks = []
    net_segment_ids = []
    net_label_ids = []

    for (ex_index, example) in enumerate(net_examples):
        net_input_id, net_input_mask, net_segment_id, net_label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, net_label_map)
        net_input_ids.append(net_input_id)
        net_input_masks.append(net_input_mask)
        net_segment_ids.append(net_segment_id)
        net_label_ids.append(net_label_id)

    net_input_ids = torch.tensor(net_input_ids)
    net_input_masks = torch.tensor(net_input_masks)
    net_segment_ids = torch.tensor(net_segment_ids)
    net_label_ids = torch.tensor(net_label_ids)

    net_teset = torch.utils.data.TensorDataset(net_input_ids, net_input_masks, net_segment_ids, net_label_ids)

    net_teloader = torch.utils.data.DataLoader(net_teset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)

    print('Preparing ssh evaluating data...')

    ssh_task = get_task('aug', args.aug_dataroot)

    ssh_examples = ssh_task.get_dev_examples()

    ssh_label_list = ssh_task.get_labels()
    ssh_label_map = {label: i for i, label in enumerate(ssh_label_list)}

    ssh_input_ids = []
    ssh_input_masks = []
    ssh_segment_ids = []
    ssh_label_ids = []

    for (ex_index, example) in enumerate(ssh_examples):
        ssh_input_id, ssh_input_mask, ssh_segment_id, ssh_label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, ssh_label_map)
        ssh_input_ids.append(ssh_input_id)
        ssh_input_masks.append(ssh_input_mask)
        ssh_segment_ids.append(ssh_segment_id)
        ssh_label_ids.append(ssh_label_id)

    ssh_input_ids = torch.tensor(ssh_input_ids)
    ssh_input_masks = torch.tensor(ssh_input_masks)
    ssh_segment_ids = torch.tensor(ssh_segment_ids)
    ssh_label_ids = torch.tensor(ssh_label_ids)

    ssh_teset = torch.utils.data.TensorDataset(ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids)

    ssh_teloader = torch.utils.data.DataLoader(ssh_teset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    return net_teset, net_teloader, ssh_teset, ssh_teloader


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
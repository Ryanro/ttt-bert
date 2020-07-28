#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/22 3:04 AM
# @Author: Zechen Li
# @File  : train_helpers.py
import torch
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from glue.tasks import get_task
from utils.utils import *


def build_model(args):
    print('Building model...')

    config = AutoConfig.from_pretrained(
        args.model,
        num_labels=2,
    )

    # create the encoders
    net = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=args.model,
        config=config
    )

    ssh = net

    fc_features = ssh.classifier.in_features

    ssh.classifier = nn.Linear(fc_features, 4)

    head = ssh.classifier

    net = torch.nn.DataParallel(net)

    ssh = torch.nn.DataParallel(ssh)

    return net, head, ssh


def prepare_train_data(args):
    print('Preparing net training data...')

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    net_task = get_task(args.task_name, args.dataroot)

    net_examples = net_task.get_train_examples()

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

    net_input_ids = torch.Tensor(net_input_ids)
    net_input_masks = torch.Tensor(net_input_masks)
    net_segment_ids = torch.Tensor(net_segment_ids)
    net_label_ids = torch.Tensor(net_label_ids)

    print('Preparing ssh training data...')

    ssh_task = get_task('aug', args.aug_dataroot)

    ssh_examples = ssh_task.get_train_examples()

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

    ssh_input_ids = torch.Tensor(ssh_input_ids)
    ssh_input_masks = torch.Tensor(ssh_input_masks)
    ssh_segment_ids = torch.Tensor(ssh_segment_ids)
    ssh_label_ids = torch.Tensor(ssh_label_ids)

    trset = torch.utils.data.TensorDataset(net_input_ids, net_input_masks, net_segment_ids, net_label_ids,
                                           ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids)

    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return trloader


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30)) *10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    net_input_ids = torch.Tensor(net_input_ids)
    net_input_masks = torch.Tensor(net_input_masks)
    net_segment_ids = torch.Tensor(net_segment_ids)
    net_label_ids = torch.Tensor(net_label_ids)

    net_teset = torch.utils.data.TensorDataset(net_input_ids, net_input_masks, net_segment_ids, net_label_ids)

    net_teloader = torch.utils.data.DataLoader(net_teset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)

    print('Preparing net evaluating data...')

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

    ssh_input_ids = torch.Tensor(ssh_input_ids)
    ssh_input_masks = torch.Tensor(ssh_input_masks)
    ssh_segment_ids = torch.Tensor(ssh_segment_ids)
    ssh_label_ids = torch.Tensor(ssh_label_ids)

    ssh_teset = torch.utils.data.TensorDataset(ssh_input_ids, ssh_input_masks, ssh_segment_ids, ssh_label_ids)

    ssh_teloader = torch.utils.data.DataLoader(ssh_teset, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)
    return net_teloader, ssh_teloader


def plot_epochs(all_err_cls, all_err_ssh, fname):
    plt.plot(np.asarray(all_err_cls) * 100, color='r', label='supervised')
    plt.plot(np.asarray(all_err_ssh) * 100, color='b', label='self-supervised')
    plt.xlabel('epoch')
    plt.ylabel('test error (%)')
    plt.legend()
    plt.savefig(fname)
    plt.close()




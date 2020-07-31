#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/28 12:19 AM
# @Author: Zechen Li
# @File  : adapt_helpers.py
from utils.train_helpers import *
import csv
from eda import *


def prepare_adapt_data(args):
    print('Preparing adaption data...')

    dataroot = os.path.join(args.dataroot, 'dev.tsv')

    with open(dataroot, "r", encoding='utf-8') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        next(reader)
        adapt_teset = []
        for line in reader:
            adapt_teset.append(line)

    return adapt_teset


def test_single(model, sentence, label, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    aug_sentences = eda(sentence, alpha=0.15, num_aug=1, method='sr')
    aug_sen = aug_sentences[0]
    tokens_a = tokenizer.tokenize(aug_sen)

    if len(tokens_a) > args.max_seq_length - 2:
        tokens_a = tokens_a[:(args.max_seq_length - 2)]

    input_ids = []
    input_masks = []
    segment_ids = []

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_id = [0] * len(tokens)
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    padding = [0] * (args.max_seq_length - len(input_id))
    input_id += padding
    input_mask += padding
    segment_id += padding

    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    segment_ids = torch.tensor(segment_ids)

    model.eval()
    input_ids, input_masks, segment_ids = \
        input_ids.cuda(), input_masks.cuda(), segment_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids, )
        outputs = nn.functional.normalize(outputs[0], dim=1)
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label].item()
    correctness = 1 if predicted.item() == label else 0
    return correctness, confidence


def adapt_single(model, sentence, optimizer, criterion, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datalists = []
    for i in range(args.batch_size):
        datalist = []
        method_label = np.random.randint(0, 4, 1)[0]
        method = augment_single_with_label(method_label)
        aug_sentence = eda(sentence, alpha=0.15, num_aug=1, method=method)[0]
        datalist.append(aug_sentence)
        datalist.append(method)
        datalists.append(datalist)

    task = get_task('aug', datalists)
    examples = task.get_aug_examples()
    label_list = task.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids = []
    input_masks = []
    segment_ids = []
    label_ids = []

    for (ex_index, example) in enumerate(examples):
        input_id, input_mask, segment_id, label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, label_map)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        label_ids.append(label_id)

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    segment_ids = torch.tensor(segment_ids)
    label_ids = torch.tensor(label_ids)

    model.train()
    for iteration in range(args.niter):
        optimizer.zero_grad()
        input_ids, input_masks, segment_ids, label_ids = \
            input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
        outputs = model(input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids, )
        outputs = nn.functional.normalize(outputs[0], dim=1)
        loss = criterion(outputs, label_ids)
        loss.backward()
        optimizer.step()


def test_net_single(model, sen1, sen2, label, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens_a = tokenizer.tokenize(sen1)
    tokens_b = tokenizer.tokenize(sen2)
    truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)

    input_ids = []
    input_masks = []
    segment_ids = []

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_id = [0] * len(tokens)

    tokens += tokens_b + ["[SEP]"]
    segment_id += [1] * (len(tokens_b) + 1)

    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    padding = [0] * (args.max_seq_length - len(input_id))
    input_id += padding
    input_mask += padding
    segment_id += padding

    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

    task = get_task(args.task_name, './')

    label_list = task.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}
    label_id = label_map[label]

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    segment_ids = torch.tensor(segment_ids)

    model.eval()
    input_ids, input_masks, segment_ids = \
        input_ids.cuda(), input_masks.cuda(), segment_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids=input_ids,
                        attention_mask=input_masks,
                        token_type_ids=segment_ids, )
        outputs = nn.functional.normalize(outputs[0], dim=1)
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[label_id].item()
    correctness = 1 if predicted.item() == label_id else 0
    return correctness, confidence


def trerr_single(model, sentence, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datalists = []
    for i in range(4):
        datalist = []
        method = augment_single_with_label(i)
        aug_sentence = eda(sentence, alpha=0.15, num_aug=1, method=method)[0]
        datalist.append(aug_sentence)
        datalist.append(method)
        datalists.append(datalist)

    task = get_task('aug', datalists)
    examples = task.get_aug_examples()
    label_list = task.get_labels()
    label_map = {label: i for i, label in enumerate(label_list)}

    input_ids = []
    input_masks = []
    segment_ids = []
    label_ids = []

    for (ex_index, example) in enumerate(examples):
        input_id, input_mask, segment_id, label_id = \
            convert_example_to_feature(example, tokenizer, args.max_seq_length, label_map)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        label_ids.append(label_id)

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    segment_ids = torch.tensor(segment_ids)
    label_ids = torch.tensor(label_ids)

    model.eval()
    input_id, input_mask, segment_id, label_id = \
        input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
    with torch.no_grad():
        outputs = model(input_ids=input_id,
                        attention_mask=input_mask,
                        token_type_ids=segment_id, )
        outputs = nn.functional.normalize(outputs[0], dim=1)
        _, predicted = outputs.max(1)
    return predicted.eq(label_id).cpu()

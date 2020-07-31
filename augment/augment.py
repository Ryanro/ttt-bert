#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/25 2:55 AM
# @Author: Zechen Li
# @File  : augment.py

import pandas as pd
import numpy as np
import csv
from eda import *


aug_df = pd.DataFrame(columns=["sentence1", "label"])

with open('../RTE/train.tsv', "r", encoding='utf-8') as f:
    reader = csv.reader(f, delimiter="\t", quotechar=None)
    lines = []
    for line in reader:
        lines.append(line)


ori_df = pd.DataFrame(columns=["sentence1", "sentence2", "label"])
for (i, line) in enumerate(lines):
    if i == 0:
        continue
    text_a = line[1]
    text_b = line[2]
    label = line[3]
    ori_df = ori_df.append({'sentence1': text_a,'sentence2': text_b, 'label': label}, ignore_index=True)

print(ori_df.head())

for i in ori_df.sentence1:
    ori_sentence = i

    method_label = np.random.randint(0, 4, 1)[0]
    method = augment_single_with_label(method_label)

    aug_sentences = eda(ori_sentence, alpha=0.15, num_aug=1, method=method)
    for aug_sentence in aug_sentences:
        aug_df = aug_df.append({'sentence1': aug_sentence, 'label': method}, ignore_index=True)

print("generated augmented sentences finished.")

print(aug_df['label'].value_counts(normalize=True) * 100)

aug_df.to_csv('augment_train.tsv', sep = '\t', index=False)


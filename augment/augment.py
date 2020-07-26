#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/25 2:55 AM
# @Author: Zechen Li
# @File  : augment1.py

import pandas as pd
import numpy as np

from eda import *


def augment_single_with_label(method_label):
    if method_label == 1:
        method = 'sr'
    elif method_label == 2:
        method = 'ri'
    elif method_label == 3:
        method = 'rs'
    else:
        method = 'rd'
    return method


ori_df = pd.read_csv('../RTE/dev.tsv', sep='\t')

aug_df = pd.DataFrame(columns=["sentence", "label"])

for i in ori_df.sentence1:
    ori_sentence = i

    method_label = np.random.randint(1, 5, 1)[0]
    method = augment_single_with_label(method_label)

    aug_sentences = eda(ori_sentence, alpha=0.1, num_aug=1, method=method)
    for aug_sentence in aug_sentences:
        aug_df = aug_df.append({'sentence': aug_sentence, 'label': method}, ignore_index=True)

print("generated augmented sentences finished.")

print(aug_df['label'].value_counts(normalize=True) * 100)

aug_df.to_csv('augment_dev.tsv', sep = '\t', index=False)


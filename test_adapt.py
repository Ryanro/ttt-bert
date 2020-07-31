#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2020/7/28 12:22 AM
# @Author: Zechen Li
# @File  : test_adapt.py

from __future__ import print_function
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from utils.adapt_helpers import *
from utils.test_helpers import *
from utils.train_helpers import *
from sklearn.metrics import accuracy_score, matthews_corrcoef


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./RTE/')
parser.add_argument('--aug_dataroot', default='./augment_data/RTE')
parser.add_argument('--task_name', default='RTE')
########################################################################
parser.add_argument('--model', default='nghuyong/ernie-2.0-large-en')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--workers', default=16, type=int)
########################################################################
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_seq_length', default=128, type=int)
parser.add_argument('--niter', default=10, type=int)
parser.add_argument('--online', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--threshold', default=1, type=float)
########################################################################
parser.add_argument('--resume', default=None)
parser.add_argument('--outf', default='.')

args = parser.parse_args()
args.threshold += 0.001  # to correct for numeric errors
my_makedir(args.outf)

cudnn.benchmark = True

net, head, ssh = build_model(args)
adapt_teset = prepare_adapt_data(args)
net_teset, net_teloader, ssh_teset, ssh_teloader = prepare_test_data(args)

print('Resuming from %s...' % args.resume)
ckpt = torch.load('%s/ckpt.pth' % args.resume)
if args.online:
    net.load_state_dict(ckpt['net'])
    head.load_state_dict(ckpt['head'])

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(ssh.parameters(), lr=args.lr)

print('Adaption Running...')
if args.shuffle:
    random.shuffle(adapt_teset)

correct = []
sshconf = []
trerror = []
for i in tqdm(range(1, len(adapt_teset) + 1)):
    if not args.online:
        net.load_state_dict(ckpt['net'])
        head.load_state_dict(ckpt['head'])

    index, sen1, sen2, label = adapt_teset[i - 1]
    correctness_ssh, confidence_ssh = test_single(ssh, sen1, 0, args)
    sshconf.append(confidence_ssh)
    print('ssh correctness:', correctness_ssh)
    correctness_net1, confidence_net1 = test_net_single(net, sen1, sen2, label, args)
    print('net correctness1:', correctness_net1)
    if sshconf[-1] < args.threshold:
        print("Doing adaption...")
        adapt_single(ssh, sen1, optimizer, criterion, args)
    correctness_net2, confidence_net2 = test_net_single(net, sen1, sen2, label, args)
    correct.append(correctness_net2)
    print('net correctness2:', correctness_net2)
    trerror.append(trerr_single(ssh, sen1, args))

print('Adapted test error cls %.2f' % ((1 - mean(correct)) * 100))
rdict = {'cls_correct': np.asarray(correct), 'ssh_confide': np.asarray(sshconf),
         'cls_adapted': 1 - mean(correct), 'trerror': trerror}
torch.save(rdict, args.outf + 'ada.pth')

# Validation

# Put model in evaluation mode to evaluate loss on the validation set
net.eval()

# Tracking variables
eval_accuracy, eval_mcc_accuracy, nb_eval_steps = 0, 0, 0

# Evaluate data for one epoch
for i, (input_ids, input_masks, segment_ids, label_ids) in enumerate(net_teloader):
    with torch.no_grad():
        input_ids, input_masks, segment_ids, label_ids = \
            input_ids.cuda(), input_masks.cuda(), segment_ids.cuda(), label_ids.cuda()
        logits = net(input_ids=input_ids.long(),
                     attention_mask=input_masks.long(),
                     token_type_ids=segment_ids.long(), )

    # Move logits and labels to CPU
    logits = logits[0].to('cpu').numpy()
    label_ids = label_ids.to('cpu').numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()
    labels_flat = label_ids.flatten()
    tmp_eval_accuracy = accuracy_score(labels_flat, pred_flat)
    tmp_eval_mcc_accuracy = matthews_corrcoef(labels_flat, pred_flat)

    eval_accuracy += tmp_eval_accuracy
    eval_mcc_accuracy += tmp_eval_mcc_accuracy
    nb_eval_steps += 1

print(F'\n\tValidation Accuracy: {eval_accuracy / nb_eval_steps}')
print(F'\n\tValidation MCC Accuracy: {eval_mcc_accuracy / nb_eval_steps}')

torch.save(net.state_dict(), args.outf + 'net.pth')

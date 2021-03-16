from data_utils import collate
from emoji_data import EmojiData
from gyafc_data import GYAFCData
from meld_data import MELDData
from sst_data import SSTData
from model import BertMultiTask
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import get_data, get_n_classes, get_task_name
from transformers import BertTokenizer
import argparse
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--branch', type=int, default=0)
    parser.add_argument('--task', type=int)
    parser.add_argument('--train_tasks', nargs='+', type=int)
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    test_data = get_data(args.task, 'test')
    test_loader = DataLoader(
        test_data,
        args.batch_size,
        collate_fn=lambda x: collate(tokenizer, x),
        pin_memory=True
    )
    
    model = BertMultiTask([get_n_classes(t) for t in args.train_tasks])
    model.load_state_dict(torch.load(args.ckpt))
    model = model.cuda().eval()

    all_correct = correct = total = 0
    tp = torch.zeros(get_n_classes(args.task))
    fp = torch.zeros(get_n_classes(args.task))
    fn = torch.zeros(get_n_classes(args.task))

    for inputs, labels in tqdm(test_loader):
        inputs = {key: inputs[key].cuda() for key in inputs}

        with torch.no_grad():
            logits = model(inputs, args.branch).cpu()
        
        if args.task == 5:
            correct += ((logits > 0) == labels.bool()).float().mean(dim=1).sum()
            all_correct += ((logits > 0) == labels.bool()).all(dim=1).sum()
            tp += ((logits > 0) & labels.bool()).sum(dim=0)
            fp += ((logits > 0) & ~labels.bool()).sum(dim=0)
            fn += ((logits <= 0) & labels.bool()).sum(dim=0)
        else:
            for p, l in zip(logits.argmax(dim=1), labels):
                if p == l:
                    correct += 1
                    tp[p] += 1
                else:
                    fp[p] += 1
                    fn[l] += 1
        total += labels.shape[0]

    name = get_task_name(args.task)
    acc = correct / total * 100
    f1 = (tp / (tp + (fp + fn) / 2)).mean() * 100
    str_out = f'{name} Test: Acc {acc:.2f} F1 {f1.item():.2f}'
    if args.task == 5:
        acc_sub = all_correct / total * 100
        str_out += f' AccSub {acc_sub:.2f}'
    print(str_out)

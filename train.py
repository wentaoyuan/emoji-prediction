from data_utils import MultiTaskDataset, DistMultiTaskBatchSampler, collate
from emoji_data import EmojiData
from gyafc_data import GYAFCData
from meld_data import MELDData
from sst_data import SSTData
from model import BertMultiTask, get_loss
from tensorboardX import SummaryWriter
from time import time
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def get_data(task_id, split):
    if task_id == 0:
        return SSTData(split)
    elif task_id == 1:
        return GYAFCData(split)
    elif task_id == 2:
        return MELDData(split)
    elif task_id == 3:
        return MELDData(split, dyadic=True)
    elif task_id == 4:
        return EmojiData('PBMC', split)
    elif task_id == 5:
        return EmojiData('PBML', split)
    elif task_id == 6:
        return EmojiData('ABMC', split)


def get_n_classes(task_id):
    if task_id == 0 or task_id == 1:
        return 2
    if task_id == 2 or task_id == 3:
        return 7
    else:
        return 64


def get_task_name(task_id):
    if task_id == 0:
        return 'SST'
    elif task_id == 1:
        return 'GYAFC'
    elif task_id == 2:
        return 'MELD'
    elif task_id == 3:
        return 'MELD-Dyadic'
    elif task_id == 4:
        return 'PBMC'
    elif task_id == 5:
        return 'PBML'
    elif task_id == 6:
        return 'ABMC'


def train(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    train_data = MultiTaskDataset([get_data(t, 'train') for t in args.tasks])
    train_sampler = DistMultiTaskBatchSampler(
        train_data, args.batch_size, drop_last=True,
        rank=rank, world_size=world_size
    )
    train_loader = DataLoader(
        train_data,
        batch_sampler=train_sampler,
        collate_fn=lambda x: collate(tokenizer, x, args.max_length),
        pin_memory=True
    )
    dev_data = MultiTaskDataset([get_data(t, 'dev') for t in args.tasks])
    dev_sampler = DistMultiTaskBatchSampler(
        dev_data, args.batch_size, drop_last=False,
        rank=rank, world_size=world_size
    )
    dev_loader = DataLoader(
        dev_data,
        batch_sampler=dev_sampler,
        collate_fn=lambda x: collate(tokenizer, x),
        pin_memory=True
    )

    model = BertMultiTask(
        [get_n_classes(t) for t in args.tasks],
        [get_loss(t) for t in args.tasks]
    ).cuda()
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.Adamax(model.parameters(), args.lr)
    step = 0
    if rank == 0:
        writer = SummaryWriter(args.log_dir)

    for epoch in range(args.n_epoch):
        model.train()
        batch_time = data_time = total_loss = 0
        start = time()
        for b, (inputs, labels, task_id) in enumerate(train_loader):
            step += 1
            inputs = {key: inputs[key].cuda() for key in inputs}
            labels = labels.cuda()
            data_time += time() - start
            
            logits, loss = model(inputs, task_id, labels)
            dist.reduce(loss, 0)
            total_loss += loss.item() / world_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = torch.zeros(len(args.tasks)).cuda()
            acc_sub = torch.tensor(0).cuda()
            task = torch.zeros(len(args.tasks)).cuda()
            task[task_id] = 1
            with torch.no_grad():
                if args.tasks[task_id] == 5:
                    correct = (logits > 0) == labels.bool()
                    acc[task_id] = correct.float().mean() * 100
                    acc_sub = correct.all(dim=1).float().mean() * 100
                else:
                    acc[task_id] = (logits.argmax(dim=1) == labels).float().mean() * 100
            dist.reduce(acc, 0)
            dist.reduce(acc_sub, 0)
            dist.reduce(task, 0)

            if rank == 0:
                writer.add_scalar(f'train/loss', loss.item())
                for i, t in enumerate(args.tasks):
                    if task[i] > 0:
                        name = get_task_name(t)
                        writer.add_scalar(
                            f'train/{name}_acc', acc[i] / task[i] * 100, step
                        )
                        if t == 5:
                            writer.add_scalar(
                                f'train/{name}_acc_sub', acc_sub / task[i] * 100, step
                            )

                if (b + 1) % args.print_freq == 0:
                    print(
                        f'Epoch {epoch+1} Train: {(b+1):05d}/{len(train_loader):05d} ' \
                        f'Batch {(batch_time/(b+1)):.3f}s Data {(data_time/(b+1)):.3f}s ' \
                        f'Loss {(total_loss/(b+1)):.4f}'
                    )

            batch_time += time() - start
            start = time()

        model.eval()
        batch_time = data_time = 0
        all_correct = torch.tensor(0).cuda()
        correct = torch.zeros(len(args.tasks)).cuda()
        total = torch.zeros(len(args.tasks)).cuda()
        tp = [torch.zeros(get_n_classes(t)).cuda() for t in args.tasks]
        fp = [torch.zeros(get_n_classes(t)).cuda() for t in args.tasks]
        fn = [torch.zeros(get_n_classes(t)).cuda() for t in args.tasks]
        start = time()
        for inputs, labels, task_id in dev_loader:
            inputs = {key: inputs[key].cuda() for key in inputs}
            labels = labels.cuda()
            data_time += time() - start

            with torch.no_grad():
                logits = model(inputs, task_id)
            
                if args.tasks[task_id] == 5:
                    correct[task_id] += ((logits > 0) == labels.bool()).float().mean(dim=1).sum()
                    all_correct += ((logits > 0) == labels.bool()).all(dim=1).sum()
                    tp[task_id] += ((logits > 0) & labels.bool()).sum(dim=0)
                    fp[task_id] += ((logits > 0) & ~labels.bool()).sum(dim=0)
                    fn[task_id] += ((logits <= 0) & labels.bool()).sum(dim=0)
                else:
                    for p, l in zip(logits.argmax(dim=1), labels):
                        if p == l:
                            correct[task_id] += 1
                            tp[task_id][p] += 1
                        else:
                            fp[task_id][p] += 1
                            fn[task_id][l] += 1
                total[task_id] += labels.shape[0]
            
            batch_time += time() - start
            start = time()

        dist.reduce(all_correct, 0)
        dist.reduce(correct, 0)
        dist.reduce(total, 0)
        for tpl, fpl, fnl in zip(tp, fp, fn):
            dist.reduce(tpl, 0)
            dist.reduce(fpl, 0)
            dist.reduce(fnl, 0)

        if rank == 0:
            l = len(dev_loader)
            str_out = f'Epoch {epoch+1}  Dev : {l:05d}/{l:05d} ' \
                      f'Batch {(batch_time/l):.3f}s Data {(data_time/l):.3f}s '
            for i, t in enumerate(args.tasks):
                name = get_task_name(t)
                acc = correct[i] / total[i] * 100
                writer.add_scalar(f'dev/{name}_acc', acc, step)
                f1 = (tp[i] / (tp[i] + (fp[i] + fn[i]) / 2)).mean() * 100
                writer.add_scalar(f'dev/{name}_f1', f1, step)
                if t == 5:
                    acc_sub = all_correct / total[i] * 100
                    writer.add_scalar(f'dev/{name}_acc_sub', acc_sub, step)
                    str_out += f'{name} Acc {acc_sub.item():.2f} F1 {f1.item():.2f} '
                else:
                    str_out += f'{name} Acc {acc.item():.2f} F1 {f1.item():.2f} '
            print(str_out)
    
    if rank == 0:
        torch.save(model.module.state_dict(), f'{args.log_dir}/epoch_{epoch+1}.pth')

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--tasks', nargs='+', type=int)
    parser.add_argument('--n_gpu', type=int, default=2)
    parser.add_argument('--port', default='12345')
    # Training
    parser.add_argument('--log_dir', default='logs/test')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=196)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--n_epoch', type=int, default=7)
    parser.add_argument('--print_freq', type=int, default=100)
    args = parser.parse_args()

    mp.spawn(
        train,
        args=(args.n_gpu, args),
        nprocs=args.n_gpu,
        join=True
    )

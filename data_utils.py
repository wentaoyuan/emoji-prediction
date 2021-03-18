from math import ceil, floor
from torch.utils.data import Dataset, Sampler
import torch
import torch.distributed as dist


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        task_id, sample_id = idx
        text, labels = self.datasets[task_id][sample_id]
        return text, labels, task_id


class DistMultiTaskBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, rank=0, world_size=1):
        self.data = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        if self.drop_last:
            self.n_batches = [floor(len(d) / self.batch_size) for d in self.data.datasets]
        else:
            self.n_batches = [ceil(len(d) / self.batch_size) for d in self.data.datasets]

    def __len__(self):
        return sum(self.n_batches) // self.world_size

    def __iter__(self):
        batch_ids = torch.arange(sum(self.n_batches)).cuda()
        sample_ids = [torch.arange(len(d)).cuda() for d in self.data.datasets]
        if self.rank == 0:
            batch_ids = torch.randperm(sum(self.n_batches)).cuda()
            sample_ids = [torch.randperm(len(d)).cuda() for d in self.data.datasets]
        dist.broadcast(batch_ids, 0)
        for i in range(len(sample_ids)):
            dist.broadcast(sample_ids[i], 0)
        batches = []
        for i, s in enumerate(sample_ids):
            for j in range(0, len(s), self.batch_size):
                if self.drop_last and j + self.batch_size > len(s):
                    break
                ids = s[j:j+self.batch_size].cpu()
                batches.append(torch.stack([torch.ones(len(ids)).long() * i, ids], dim=1))
        n_batches = sum(self.n_batches) // self.world_size * self.world_size
        for i in range(self.rank, n_batches, self.world_size):
            yield batches[batch_ids[i]]


def collate(tokenizer, batch, max_length=0):
    batch = tuple(zip(*batch))
    if len(batch) == 3:
        text, labels, task_ids = batch
    else:
        text, labels = batch
    if max_length == 0:
        max_length = tokenizer.model_max_length
    encoded_input = tokenizer(
        text, max_length=max_length, padding=True,
        truncation=True, return_tensors='pt'
    )
    if len(batch) == 3:
        return encoded_input, torch.stack(labels), task_ids[0]
    else:
        return encoded_input, torch.stack(labels)

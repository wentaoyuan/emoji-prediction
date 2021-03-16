from torch.utils.data import Dataset
import torch


class GYAFCData(Dataset):
    def __init__(self, split):
        if split == 'dev':
            split = 'tune'
        self.data = []
        for cat in ['Entertainment_Music', 'Family_Relationships']:
            for l, label in enumerate(['informal', 'formal']):
                path = f'data/GYAFC_Corpus/{cat}/{split}/{label}'
                with open(path, encoding='utf-8') as f:
                    for line in f.readlines():
                        s = line.strip()
                        self.data.append((line.strip(), l))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        return text, torch.tensor(label)

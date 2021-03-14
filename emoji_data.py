from torch.utils.data import Dataset
import csv
import torch


def read_csv(csvpath):
    pbmc = {
        'train': [],
        'dev': [],
        'test': []
    }
    pbml = {
        'train': [],
        'dev': [],
        'test': []
    }
    abmc = {
        'train': [],
        'dev': [],
        'test': []
    }
    data = {}
    labels = []
    with open(csvpath, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for line in reader:
            s, label, aspect, split = line
            if label not in labels:
                labels.append(label)
            if s in data:
                if label not in data[s]['label']:
                    data[s]['aspect'].append(aspect)
                    data[s]['label'].append(label)
            else:
                data[s] = {
                    'aspect': [aspect],
                    'label': [label],
                    'split': split
                }
    for s in data:
        for aspect, label in zip(data[s]['aspect'], data[s]['label']):
            abmc[data[s]['split']].append((aspect, label))
        if len(data[s]['label']) == 1:
            pbmc[data[s]['split']].append(
                (s, data[s]['label'][0])
            )
        else:
            pbml[data[s]['split']].append(
                (s, data[s]['label'])
            )
    emojis = {emoji: i for i, emoji in enumerate(labels)}
    return pbmc, pbml, abmc, emojis


class EmojiData(Dataset):
    def __init__(self, anno, split, csvpath='data/multi_class.csv'):
        pbmc, pbml, abmc, emojis = read_csv(csvpath)
        self.anno = anno
        if anno == 'PBMC':
            self.data = pbmc[split]
        elif anno == 'PBML':
            self.data = pbml[split]
        elif anno == 'ABMC':
            self.data = abmc[split]
        self.emojis = emojis
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, emoji = self.data[index]
        if self.anno == 'ABMC':
            text = ' '.join(text.split(';'))
        if self.anno == 'PBML':
            label = torch.zeros(len(self.emojis))
            for e in emoji:
                label[self.emojis[e]] = 1
        else:
            label = torch.tensor(self.emojis[emoji])
        return text, label


if __name__ == "__main__":
    pbmc, pbml, abmc, emojis = read_csv('data/multi_class.csv')
    print(emojis)

    print('PBMC')
    total = 0
    for split in ['train', 'dev', 'test']:
        total += len(pbmc[split])
        print(split, len(pbmc[split]))
    print('total', total)

    print('PBML')
    total = 0
    for split in ['train', 'dev', 'test']:
        total += len(pbml[split])
        print(split, len(pbml[split]))
    print('total', total)

    print('ABMC')
    total = 0
    for split in ['train', 'dev', 'test']:
        total += len(abmc[split])
        print(split, len(abmc[split]))
    print('total', total)

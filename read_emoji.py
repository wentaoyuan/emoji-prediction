import csv
from torch.utils.data import Dataset
from transformers import BertTokenizer


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
    sentences = {}
    labels = []
    with open(csvpath, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for line in reader:
            s, label, aspect, split = line
            if label not in labels:
                labels.append(label)
            if s in sentences:
                if label not in sentences[s]['label']:
                    sentences[s]['label'].append(label)
            else:
                sentences[s] = {
                    'label': [label],
                    'split': split
                }
            abmc[split].append((aspect, label))
    for s in sentences:
        if len(sentences[s]['label']) == 1:
            pbmc[sentences[s]['split']].append(
                (s, sentences[s]['label'][0])
            )
        else:
            pbml[sentences[s]['split']].append(
                (s, sentences[s]['label'])
            )
    emojis = {emoji: i for i, emoji in enumerate(labels)}
    return pbmc, pbml, abmc, emojis


class EmojiData(Dataset):
    def __init__(self, anno, split, csvpath='data/multi_class.csv'):
        pbmc, pbml, abmc, emojis = read_csv(csvpath)
        self.anno = anno
        if anno == 'PMBC':
            self.data = pbmc[split]
        elif anno == 'PMBL':
            self.data = pbml[split]
        elif anno == 'AMBC':
            self.data = abmc[split]
        self.emojis = emojis
        self.tokentizer = BertTokenizer.from_pretrained('bert-large-cased')
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, emoji = self.data[index]
        if self.anno == 'ABMC':
            text = self.tokentizer(
                text.split(';'),
                is_split_into_words=True,
                return_tensors='pt'
            )
        else:
            text = self.tokentizer(text, return_tensors='pt')
        if self.anno == 'PBML':
            label = torch.zeros(len(self.emojis)).long()
            for e in emoji:
                label[self.emojis[e]] = 1
        else:
            label = self.emojis[emoji]
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

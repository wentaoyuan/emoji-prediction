from torch.utils.data import Dataset
import pickle
import torch


def sentiment_convert(val):
    if val <= 0.4:
        return 0
    elif val > 0.6:
        return 1
    else:
        return -1


class SSTData(Dataset):
    def __init__(self, split):
        if split == 'dev':
            split = 'valid'
        rating = pickle.load(open(f'data/SST2-Sentence/id_rating_{split}.p', 'rb'))
        sentence = pickle.load(open('data/SST2-Sentence/id_sentence.p', 'rb'))
        self.data = []
        for index, sentence in sentence.items():
            if index in rating.keys():
                label = sentiment_convert(rating[index])
                if label >= 0:
                    self.data.append((' '.join(sentence), label))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        return text, torch.tensor(label)
        
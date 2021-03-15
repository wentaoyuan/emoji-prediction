from torch.utils.data import Dataset
import csv
import torch


class MELDData(Dataset):
    def __init__(self, split, dyadic=False):
        emotion_to_label = {
            'anger': 0, 'disgust': 1, 'sadness': 2, 'joy': 3,
            'neutral': 4, 'surprise': 5, 'fear': 6
        }
        if dyadic:
            csvpath = f'data/MELD_Dyadic/{split}_sent_emo_dya.csv'
            sen_id = 0
            emo_id = 2
        else:
            csvpath = f'data/MELD/{split}_sent_emo.csv'
            sen_id = 1
            emo_id = 3
        self.data = []
        with open(csvpath, encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for row in reader:
                text = row[sen_id]
                label = emotion_to_label[row[emo_id]]
                self.data.append((text, label))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        return text, torch.tensor(label)

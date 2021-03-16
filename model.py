from transformers import BertModel
import torch
import torch.nn as nn


def get_loss(task_id):
    if task_id == 5:
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()


class BertMultiTask(nn.Module):
    def __init__(self, n_classes, losses=None):
        super(BertMultiTask, self).__init__()
        self.backbone = BertModel.from_pretrained("bert-large-cased")
        for i, n in enumerate(n_classes):
            setattr(self, f'fc{i}', nn.Linear(1024, n))
        self.losses = losses

    def forward(self, inputs, task_id, labels=None):
        output = self.backbone(**inputs).pooler_output
        logits = getattr(self, f'fc{task_id}')(output)
        if labels is not None:
            return logits, self.losses[task_id](logits, labels)
        else:
            return logits

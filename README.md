# Implementation of Multi-resolution Annotations for Emoji Prediction

## Dependencies:
[Pytorch](https://pytorch.org) with CUDA and [🤗 Transformers](https://huggingface.co/transformers) library.

Example setup with conda:
1. `conda create -n transformer python`
2. `conda install pytorch cudatoolkit=10.1 -c pytorch -c conda-forge`
3. `pip install transformers`
​
## Data preparation
### [Stanford Sentiment Treebank(SST)](https://nlp.stanford.edu/sentiment/index.html)
As the raw data only provides the label corresponding to each phrase rather than each sentence. Additional pre-processing is required to access the label for each sentence. So, we use this package (https://github.com/frankaging/SST2-Sentence) that allows us to directly load all those files using pickle module in python. As all the raw data for SST are stored in `/root` directory of the package, we cloned this github folder and put our SST processing code under `/SST2-Sentence` directory that directly pickle the loaded data for each split set and format it into desired Pytorch dataloader class setting.
​
### [MELD/MELD Dyadic](https://affective-meld.github.io)
The datasets are downloaded from the original website but for convenience, they can be accessed in this [Google Drive folder](https://drive.google.com/drive/folders/1XONpNbPa5mpu6V9WVpkQwNvCD40zRWtn?usp=sharing). Our processing script is in the same directory as the dataset folder and can be directly run to load the dataset and format it into desired Pytorch dataloader class setting. 
​
### [GYAFC Corpus](https://github.com/raosudha89/GYAFC-corpus)
We contacted the original author for access of the data but it's also currently available for download in Google drive. Our processing script should be placed in the same directory as the datasets (GYAFC_Corpus) and can be directly run to load the dataset and format it into desired Pytorch dataloader class setting. 
​
### [Multi-Resolution Emoji Prediction (MREP) Dataset]()
We contacted the authors of the original paper [1] for access of the dataset through email. The data can be found in `data/multi_class.csv`. `emoji_data.py` provide util functions to load the dataset and format it into a Pytorch `Dataset` class.

## Training code + command:
`train.py` contains code to do distributed multi-GPU fine-tuning of a pretrained `bert-large-cased` model on multiple text classfication datasets. Each dataset has a unique task ID listed below:
```
SST: 0
GYAFC: 1
MELD: 2
MELD-Dyadic: 3
PBMC: 4
PBML: 5
ABMC: 6
```
For example, to fine-tune jointly on SST and PBMC with 2 GPUs
```
python train.py --log logs/sst_pbmc --n_gpu 2 --tasks 0 4
```

## Evaluation code + command:
​`test.py` includes code which calculates average accuracy and F-1 score on the test set. For example,
```
python test.py --ckpt logs/sst_pbmc/model.pth --task 0 --train_tasks 0 4
```
loads a model fine-tuned jointly on SST and PBMC and test it on SST's test set. Note that the task IDs specified by `--train_tasks` should match the training tasks for the checkpoint specified by `--ckpt`.
​
## Table of results:
​We reproduced part of the experiment results in Table 5 of [1], shown below.
|             |  SST  |       | GYAFC |       |  MELD |       | MELD-Dyadic |       |
|:-----------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:-----:|
|             |  ACC  |  F-1  |  ACC  |  F-1  |  ACC  |  F-1  |     ACC     |  F-1  |
| Single-task | 93.19 | 93.18 | 92.60 | 92.56 | 61.57 | 45.15 |    59.91    | 43.87 |
|    +PBMC    | 90.12 | 90.11 | 93.09 | 93.05 | 60.84 | 43.91 |    59.62    | 42.33 |
|    +PBML    | 94.24 | 94.24 | 93.28 | 93.23 | 60.88 | 43.18 |    60.85    | 45.16 |
|    +ABMC    | 92.37 | 92.37 | 93.40 | 93.35 | 59.54 | 42.56 |    59.21    | 43.25 |

## Citation
[1] Ma, Weicheng, et al. "Multi-resolution Annotations for Emoji Prediction." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.

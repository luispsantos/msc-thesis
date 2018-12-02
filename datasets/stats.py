import pandas as pd
from pathlib import Path
import argparse
from util.dataset import Dataset
from util.util import count_sents

parser = argparse.ArgumentParser(description='Print dataset statistics such as the number of tokens and sentences, as well as the size of POS and NER tagsets')
parser.add_argument('dataset_path', help='Path to a dataset containing a data folder')
parser.add_argument('--print-tag-counts', action='store_true', help='Print the counts of POS and NER tags')

args = parser.parse_args()
dataset_path, print_tag_counts = Path(args.dataset_path), args.print_tag_counts

# load dataset
dataset = Dataset(dataset_path)

# output number of tokens and sentences for train, dev and test sets
for dataset_type, data_df in dataset:
    num_sents, num_tokens = count_sents(data_df)
    print('{} - {} sents, {} tokens'.format(dataset_type.capitalize(), num_sents, num_tokens))

# compute POS and NER statistics on the training set
train_data = dataset.data['train']

if 'POS' in train_data.columns:
    pos = train_data['POS']
    pos_counts = pos.value_counts()

    num_pos = pos.nunique()
    print('Number of distinct POS tags: {}'.format(num_pos)) 

if 'NER' in train_data.columns:
    ner = train_data['NER']
    ner_counts = ner.value_counts()

    num_ner = ner.nunique()
    print('Number of distinct NER tags: {}'.format(num_ner)) 

# print counts for POS and NER tags
if print_tag_counts and 'POS' in train_data.columns:
    print('\nPOS tags and their counts')
    print(pos_counts.to_string())

if print_tag_counts and 'NER' in train_data.columns:
    print('\nNER tags and their counts')
    print(ner_counts.to_string())


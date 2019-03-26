from pathlib import Path
import pandas as pd
import re
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing util package from parent directory
sys.path.insert(1, str(Path.cwd().parent))
from util import *

# read variables from the configuration file
config = load_yaml('config.yml')
dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
rules = load_yaml('rules.yml')
ner_tagset_map = rules['ner_tagset_map']

token_re = re.compile('^(?P<Token>.+?)_NER_(?P<NER>.+)$')

# read class dictionary
class_dict = load_yaml(dataset_in_dir / 'classes.json')

# read JSONL data
data_df = pd.read_json(dataset_in_dir / 'sents.json', lines=True)

# read sentences to a DataFrame
data_df = read_data(data_df.conll, token_re)

# compute unused classes from class dictionary
unused_keys = class_dict.keys() - ner_tagset_map.keys()
unused_keys_re = '|'.join(sorted(unused_keys, key=len, reverse=True))

# remove unused classes and extra underscores
ner = data_df.NER.str.replace(f'({unused_keys_re})_?', '')
ner = ner.str.replace('_$', '')

# remove duplicate classes
ner = ner.str.replace(r'(e_\d+)_\1', r'\1')

# replace empty string (due to deleted classes) by O
ner = ner.replace('', 'O')

# map NER classes to more standard names
data_df.NER = ner.map(ner_tagset_map)

# add BIO encoding to NER column
add_bio_encoding(data_df.NER)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)

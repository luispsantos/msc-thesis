from pathlib import Path
import pandas as pd
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

# read Paramopama data
data_in_path = dataset_in_dir / 'corpus_paramopama+second_harem.txt'
data_df = read_conll(data_in_path, '\t', output_columns)

# map NER tags to more standard names
data_df.NER = data_df.NER.map(ner_tagset_map)

# add BIO encoding to NER column
add_bio_encoding(data_df.NER)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


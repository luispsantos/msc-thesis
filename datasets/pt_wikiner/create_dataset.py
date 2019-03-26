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
pos_tagset_ud_map = rules['pos_tagset_ud_map']

wikiner_file = dataset_in_dir / 'aij-wikiner-pt-wp3'
token_re = re.compile('^(?P<Token>.+?)\|(?P<POS>[A-Za-z_+]+)\|(?P<NER>[A-Z-]+)$')

# open WikiNER file and remove empty lines
with wikiner_file.open('r') as f:
    sents = [sent for sent in f if sent.strip()]

# read WikiNER data into a DataFrame
data_df = read_data(sents, token_re)

# convert tags to UD tagset
data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)

# convert IOB tagging scheme to BIO
iob_to_bio(data_df.NER)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


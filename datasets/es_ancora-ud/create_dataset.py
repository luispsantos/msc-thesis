from pathlib import Path
import pandas as pd
import yaml
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from util import *

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

multiword_upos_map = rules['multiword_upos_map']

for dataset_type in ['train', 'dev', 'test']:
    data_in_path = dataset_in_dir / f'es_ancora-ud-{dataset_type}.conllu'
    data_out_path = dataset_out_dir / f'{dataset_type}.txt'

    # read CoNLL-U data and extract multi-word tokens
    data_df = read_conllu(data_in_path)
    data_df = extract_multiwords(data_df, multiword_upos_map)

    # write data to disk
    write_data(data_df, data_out_path, output_columns)


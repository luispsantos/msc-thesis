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
multiword_upos_map = rules['multiword_upos_map']

def process_dataset(data_in_path):
    # read CoNLL-U data and extract multi-word tokens
    data_df = read_conllu(data_in_path)
    data_df = extract_multiwords(data_df, multiword_upos_map)

    # replace PART tags by ADJ
    data_df.UPOS.replace('PART', 'ADJ', inplace=True)

    return data_df

# process dataset with pre-made data splits and write data to disk
data_in_format = 'pt_bosque-ud-{dataset_type}.conllu'.format
dataset_with_splits(process_dataset, data_in_format, dataset_in_dir, dataset_out_dir, output_columns)

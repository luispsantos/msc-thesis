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

#read variables from the configuration file
config = load_yaml('config.yml')
dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
rules = load_yaml('rules.yml')
pos_tagset_ud_map, rules = rules['pos_tagset_ud_map'], rules['rules']

matcher = RuleMatcher(rules)
token_re = re.compile('^(?P<Token>.+?)_(?P<POS>[A-Z+-]+)$')

for dataset_type in ['train', 'dev', 'test']:
    data_in_path = dataset_in_dir / f'macmorpho-{dataset_type}.txt'
    data_out_path = dataset_out_dir / f'{dataset_type}.txt'

    # read sentences to a DataFrame
    with data_in_path.open('r') as f:
        sents = f.readlines()
        data_df = read_data(sents, token_re)

    # convert the POS tagset
    data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)
    data_df, rule_counts = matcher.apply_rules(data_df)

    # write data to disk
    write_data(data_df, data_out_path, output_columns)


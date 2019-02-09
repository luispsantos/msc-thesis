import pandas as pd
from pathlib import Path
import yaml
import re
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from rule_matcher import RuleMatcher
from util import *

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_separator, output_columns  = config['output_separator'], config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

rules, pos_tagset_ud_map = rules['rules'], rules['pos_tagset_ud_map']
matcher = RuleMatcher(rules)

token_re = re.compile('^(?P<Token>.+?)_(?P<POS>[A-Z+-]+)$')

def token_generator(dataset_in_path):
    for sent in dataset_in_path.open('r'):
        for token in sent.split():
            yield token
        yield SENT_BOUNDARY

for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'macmorpho-{}.txt'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    data_df = read_data(token_generator(dataset_in_path), token_re)

    # convert the POS tagset
    data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)
    data_df, rule_counts = matcher.apply_rules(data_df)

    write_data(data_df, dataset_out_path, output_separator, output_columns)
    print('Created file {}'.format(dataset_out_path))


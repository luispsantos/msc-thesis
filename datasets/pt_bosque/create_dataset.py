import pandas as pd
from pathlib import Path
import yaml
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from rule_matcher import RuleMatcher
import contractions
from util import *

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_separator, output_columns  = config['output_separator'], config['output_columns']
keep_contractions = config['keep_contractions']

# create output dataset directory if it doesn't exist
if not dataset_out_dir.exists():
    dataset_out_dir.mkdir(parents=True)

matcher = RuleMatcher(contractions.rule_list)

for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'pt_bosque-ud-{}.conllu'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    data_df = read_conllu(dataset_in_path)

    # make contractions
    if keep_contractions:
        data_df = matcher.apply_rules(data_df)

    write_data(data_df, dataset_out_path, output_separator)
    print('Created file {}'.format(dataset_out_path))

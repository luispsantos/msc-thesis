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

def process_dataset(data_in_path):
    # read sentences to a DataFrame
    sents = data_in_path.read_text().splitlines()
    data_df = read_data(sents, token_re)

    # convert the POS tagset
    data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)
    data_df, rule_counts = matcher.apply_rules(data_df)

    return data_df

# process dataset with pre-made data splits and write data to disk
data_in_format = 'macmorpho-{dataset_type}.txt'.format
dataset_with_splits(process_dataset, data_in_format, dataset_in_dir, dataset_out_dir, output_columns)

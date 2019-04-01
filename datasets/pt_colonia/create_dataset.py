from pathlib import Path
from io import StringIO
import pandas as pd
import re
import sys
import os
import csv

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
pos_tagset_ud_map, rules = rules['pos_tagset_ud_map'], rules['rules']

def preprocess_text(text):
    # remove <text> start and end tags
    text = re.sub('<text(?: id)?=".+?"?>\n', '', text)
    text = text.replace('</text>\n', '')

    # remove empty sentences
    text = text.replace('<s>\n</s>\n', '')

    # remove sentence boundary tags
    text = text.replace('<s>\n', '')
    text = text.replace('<s>..\n', '')
    text = text.replace('</s>', '')  # keep newline at sentence ends

    # remove section number tags (e.g. <118>, <161>)
    text = re.sub('<\d+>\n', '', text)
    
    return text

def text_to_dataframe(text):
    data_df = pd.read_csv(StringIO(text), sep='\t', names=['Token', 'POS', 'Lemma'],
                          usecols=[0, 1], na_values=[''], keep_default_na=False,
                          skip_blank_lines=False, quoting=csv.QUOTE_NONE)
    return data_df

data_df = read_text_files(dataset_in_dir, preprocess_text, text_to_dataframe)

# apply dataset-specific rules
matcher = RuleMatcher(rules)
data_df, rule_counts = matcher.apply_rules(data_df)

# convert tags to UD tagset
data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


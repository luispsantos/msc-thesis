from pathlib import Path
import pandas as pd
import yaml
import re
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from rule_matcher import SequentialMatcher
from historical_dataset import read_text_files
from util import *

# read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

pos_tagset_ud_map, pos_corrections = rules['pos_tagset_ud_map'], rules['pos_corrections']
nominal_tags, contractions, rules = rules['nominal_tags'], rules['contractions'], rules['rules']

token_re = re.compile('^(?P<Token>.*)/(?P<POS>[A-Za-z0-9()<>@$.,_"!+-]*)$')
matcher = SequentialMatcher(contractions, rules)

def preprocess_text(text):
    # remove the lines which specify the file format
    text = text.replace('#!FORMAT=POS_0', '')

    # remove the text and heading tags
    text = re.sub('</?text>', '', text)
    text = re.sub('<_?heading>/CODE', '', text)
    text = re.sub('<_?ml>/CODE', '', text)

    # remove more metadata
    text = re.sub('<P.+?>/CODE', '', text)
    text = text.replace('.d/CODE', '')
    text = text.replace('/CODE', '')

    # convert <paren> tags to parentheses
    text = text.replace('<paren>', '(')
    text = text.replace('<$$paren>', ')')

    return text

def text_to_dataframe(text):
    # split text into lines and remove empty lines
    sents = [sent for sent in text.splitlines() if sent.strip()]

    data_df = read_data(sents, token_re)
    return data_df
    
data_df = read_text_files(dataset_in_dir, preprocess_text, text_to_dataframe)

# remove tokens which contain either an empty token or POS tag
data_df = data_df[(data_df.POS != 'ID') & (data_df.POS != 'PONFP')]

# remove tokens which contain either an empty token or POS tag
data_df = data_df[(data_df.Token != '') & (data_df.POS != '')]

# remove digits at the end of POS tags
pos = data_df.POS.astype('category').str.replace('-\d+$', '')

# replace underscores for dashes on all POS tags as these mean the same
pos = pos.astype('category').str.replace('_', '-', regex=False)

# remove inflection information such as gender and number from nominal tags
tags_concat = '|'.join(re.escape(tag) for tag in nominal_tags)
pos = pos.astype('category').str.replace(f'^(?P<pos>{tags_concat})(?:(?P<feminine>-F)|(?P<double_gender>-G))?(?P<plural>-P)?$', '\g<pos>')

# remove inflection information such as mood and tense from verbal tags
pos = pos.astype('category').str.replace(f'^(?P<pos>SR|HV|ET|TR|VB)-(?P<inflection>F|I|P|SP|D|RA|SD|R|SR|G|PP|PP-P)', '\g<pos>')
data_df.POS = pos

# apply dataset-specific rules
data_df, rule_counts = matcher.apply_rules(data_df)

# apply some POS corrections before converting to UD tagset
replace_values(pos_corrections, data_df.POS)

# map multiple forms of indicating verbs with clitics to a single possible form
data_df.POS = data_df.POS.astype('category').str.replace(f'^(?P<pos>SR|HV|HV\+P|ET|TR|VB|VBP|ADV)((?P<separator>\+|-|!)(?P<clitic>CL|SE))+$', '\g<pos>+CL')

# convert tags to UD tagset
data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


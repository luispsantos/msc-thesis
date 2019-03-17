import pandas as pd
from pathlib import Path
import yaml
import re
import sys
import os
from io import StringIO
import csv

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from historical_dataset import read_text_files, sent_tokenize
from util import *

# read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

pos_tagset_ud_map = rules['pos_tagset_ud_map']

def preprocess_text(text):
    # URL decode quote character
    text = text.replace('&quot;', '"')

    # fix C tag (which doesn't exist) and map to V tag
    text = text.replace('CS+', 'VS+')

    # fix erroneous Fc tag concatenation
    text = text.replace('+Fc', '')

    # fix erroneous haver (auxiliar verb) as a tag concatenation
    text = text.replace('haver+PP', 'VA+PP')
    
    return text

def text_to_dataframe(text):
    data_df = pd.read_csv(StringIO(text), sep='\t', names=['Token', 'TokenStandard',
                          'POS', 'Lemma'], usecols=[0, 1, 2], quoting=csv.QUOTE_NONE)
    return data_df

def get_first_character(pos_tag):
    # obtain first character of each POS tag (e.g. VMIS3S0 -> V, SPS00+DA0FS0 -> S+D, etc.)
    return '+'.join(single_tag[:1] for single_tag in pos_tag.split('+'))

data_df = read_text_files(dataset_in_dir, preprocess_text, text_to_dataframe)

# discard tokens (or even whole texts) without a POS tag
data_df = data_df[pd.notna(data_df.POS)]

# discard large groups of consecutive tokens with the same POS tag
# for example, some texts contain a mix of tokens without POS tags
# and a large chunk of tokens all tagged the same by mistake
pos_groups = (data_df.POS != data_df.POS.shift()).cumsum()
data_df = data_df[data_df.groupby(pos_groups)['POS'].transform('size').lt(25)]

# perform sentence tokenization since Post Scriptum has no sentence boundaries
data_df = sent_tokenize(data_df, data_df.TokenStandard, 'pt_core_news_sm')

# obtain the first and first+second characters of the POS tag (e.g. VMIS3S0 -> V and VM)
# the POS tagset of Post Scriptum is position-based, where the initial positions
# carry the most relevant information to make the conversion of tagsets
pos = data_df.POS.astype('category')
data_df['POS_0'], data_df['POS_0+1'] = pos.apply(get_first_character), pos.str[:2]

# whenever Token is empty and standard Token is not (mostly occurs on punctuation),
# copy the text of standard Token to original Token to avoid existence of empty tokens
empty_token_mask = pd.isna(data_df.Token) & pd.notna(data_df.TokenStandard)
data_df.Token.mask(empty_token_mask, data_df.TokenStandard, inplace=True)

# map POS tagset to UD tagset based on the first and first+second characters of POS tags
data_df['UPOS'] = data_df['POS_0'].map(pos_tagset_ud_map)
replace_values(pos_tagset_ud_map, data_df['POS_0+1'], data_df.UPOS)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import re

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_separator, output_columns  = config['output_separator'], config['output_columns']

# create output dataset directory if it doesn't exist
if not dataset_out_dir.exists():
    dataset_out_dir.mkdir(parents=True)

token_re = re.compile('^(?P<Token>.+?)_(?P<POS>[A-Z+-]+)$')

pos_tagset_ud_map = {
    'ADJ': 'ADJ',
    'ADV': 'ADV',
    'ADV-KS': 'SCONJ',
    'ART': 'DET',
    'CUR': 'SYM',
    'IN': 'INTJ',
    'KC': 'CCONJ',
    'KS': 'SCONJ',
    'N': 'NOUN',
    'NPROP': 'PROPN',
    'NUM': 'NUM',
    'PCP': 'VERB',
    'PDEN': 'ADV',
    'PREP': 'ADP',
    'PREP+ADV': 'ADP+ADV',
    'PREP+ART': 'ADP+DET',
    'PREP+PRO-KS': 'ADP+SCONJ',
    'PREP+PROADJ': 'ADP+DET',
    'PREP+PROPESS': 'ADP+PRON',
    'PREP+PROSUB': 'ADP+PRON',
    'PRO-KS': 'SCONJ',
    'PROADJ': 'DET',
    'PROPESS': 'PRON',
    'PROSUB': 'PRON',
    'PU': 'PUNCT',
    'V': 'VERB',
}

def token_generator(dataset_in_path):
    for sent in dataset_in_path.open('r'):
        for token in sent.split():
            yield token
        # NaN values indicate sentence boundaries
        yield np.nan

for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'macmorpho-{}.txt'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    # create a Series of raw tokens by means of a token generator on the input dataset file
    raw_tokens = pd.Series(token_generator(dataset_in_path))

    # make sure the token regex matches the whole token
    assert raw_tokens.str.match(token_re).all(), 'Token regex failed to match at least one token'

    # convert a Series of raw tokens into a DataFrame of cleaned-up strings
    # where each capture group names in the regex are used as column names
    data = raw_tokens.str.extract(token_re)

    # convert the POS tagset
    data.POS = data.POS.map(pos_tagset_ud_map)

    # define output format as joining columns with an output separator (e.g., ' ', '\t')
    # in a line-based format of a token per line where empty lines denote sentence boundaries
    output_rows = data.Token + output_separator + data.POS
    output_text = output_rows.str.cat(sep='\n', na_rep='')

    dataset_out_path.write_text(output_text)
    print('Created file {}'.format(dataset_out_path))


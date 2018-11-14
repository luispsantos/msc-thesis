import numpy as np
import pandas as pd
from pathlib import Path
from lxml import etree
from html import escape
import yaml
import random
import re

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
train_proportion, dev_proportion = config['train_proportion'], config['dev_proportion']
output_separator, output_columns  = config['output_separator'], config['output_columns']

# fix the random seed for reproducibility
random.seed(123)

# create output dataset directory if it doesn't exist
if not dataset_out_dir.exists():
    dataset_out_dir.mkdir(parents=True)

token_re = re.compile(r'^(?:\\\*)?(?P<Token>.+?)(?:\*/)?(?:/(?P<Lemma>[^a-z]+))?(?:/(?P<POS>[A-Z]+)\d?)(?:#(?P<FEATS>[\w?-]+))?(?:\[(?P<NER>[A-Z-]+)\])$')

pos_tagset_ud_map = {
    'ADJ': 'ADJ',
    'ADV': 'ADV',
    'CARD': 'NUM',
    'CJ': 'CCONJ',
    'CL': 'PRON',
    'CN': 'NOUN',
    'DA': 'DET',
    'DEM': 'PRON',
    'DFR': 'NUM',
    'DGT': 'NUM',
    'DGTR': 'NUM',
    'DM': 'INTJ',
    'EOE': 'ADV',
    'GER': 'VERB',
    'IA': 'DET',
    'IND': 'PRON',
    'INF': 'VERB',
    'INT': 'PRON',
    'ITJ': 'INTJ',
    'LADV': 'ADV',
    'LCJ': 'CCONJ',
    'LCN': 'NOUN',
    'LDEM': 'PRON',
    'LDFR': 'NUM',
    'LITJ': 'INTJ',
    'LPREP': 'ADP',
    'LPRS': 'PRON',
    'LREL': 'PRON',
    'LTR': 'SYM',
    'MGT': 'NOUN',
    'MTH': 'NOUN',
    'ORD': 'ADJ',
    'PADR': 'NOUN',
    'PNM': 'PROPN',
    'PNT': 'PUNCT',
    'POSS': 'DET',
    'PP': 'ADV',
    'PPA': 'ADJ',
    'PPT': 'VERB',
    'PREP': 'ADP',
    'PRS': 'PRON',
    'QNT': 'DET',
    'REL': 'PRON',
    'STT': 'NOUN',
    'SYB': 'SYM',
    'UM': 'DET',
    'V': 'VERB',
    'VAUX': 'AUX',
    'WD': 'NOUN'
}

def normalize_token(token_dict):

    # remove last "_" in tokens that operate as contractions (e.g., por_, de_, em_)
    if token_dict['Token'][-1] == '_' and len(token_dict['Token']) > 1:
        token_dict['Token'] = token_dict['Token'][:-1]

    #map POS tags from CINTIL tagset to UD tagset
    token_pos_cintil = token_dict['POS']

    if token_dict['Token'] == 'sido' and token_pos_cintil == 'PPT':
        token_pos_ud = 'AUX'
    # discard tokens with optional gender and number (e.g., (s), (as), etc.)
    elif token_pos_cintil == 'TERMN':
        return False
    else:
        token_pos_ud = pos_tagset_ud_map[token_pos_cintil] if token_pos_cintil in pos_tagset_ud_map else token_pos_cintil

    token_dict['POS'] = token_pos_ud

    return token_dict


def preprocess_text(cintil_text):
    # delete unnecessary tags: <i>, </i>, <t>, </t>
    # their function is not fundamental to the creation of POS or NER systems
    # furthermore some of these tags are opened without being properly closed
    cintil_text = re.sub(r'</?(i|t)> ', '', cintil_text)

    # URL encode '&' symbol - necessary in order to use a XML parser
    cintil_text = cintil_text.replace('&', '&amp;')

    # URL encode '<' and '>' symbols whenever they occur as tokens (not as XML tags)
    cintil_text = re.sub(r'(?:\\\*)?(<|>)(?:\*/)?/[^>]+?\[.+?\]', lambda match: escape(match.group(0)), cintil_text)

    return cintil_text

def token_generator(dataset_in_path):
    # read CINTIL's file contents
    cintil_text = dataset_in_path.read_text()

    # preprocess text (required for the XML parser to work)
    cintil_text = preprocess_text(cintil_text)

    # parse XML
    cintil = etree.fromstring(cintil_text)

    # find XML tags which correspond to sentences
    sents = cintil.xpath('excerpt/text/p/s/text()')

    for sent in sents:
        for token in sent.strip().split():
            yield token
        # NaN values indicate sentence boundaries
        yield np.nan


dataset_in_path = dataset_in_dir / 'CINTIL-WRITTEN.txt'

# create a Series of raw tokens by means of a token generator on the input dataset file
raw_tokens = pd.Series(token_generator(dataset_in_path))

# make sure the token regex matches the whole token
assert raw_tokens.str.match(token_re).all(), 'Token regex failed to match at least one token'

# convert a Series of raw tokens into a DataFrame of cleaned-up strings
# where each capture group names in the regex are used as column names
data = raw_tokens.str.extract(token_re)


data = data[output_columns]
pos = data.POS

# randomize the dataset before splitting
random.shuffle(sents)

# create train-dev-test split
num_sents = len(sents)
dataset_sents = {}

train_split = int(train_proportion * num_sents)
dev_split = int((train_proportion + dev_proportion) * num_sents)

dataset_sents['train'] = sents[:train_split]
dataset_sents['dev'] = sents[train_split:dev_split]
dataset_sents['test'] = sents[dev_split:]

for dataset_type in ['train', 'dev', 'test']:
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    with dataset_out_path.open('w') as f:
        for sent_tokens in dataset_sents[dataset_type]:
            for token_dict in sent_tokens:
                token_columns = [token_dict[col] for col in output_columns]

                token_line = output_separator.join(token_columns)
                f.write(token_line + '\n')

            # write an empty line to denote sentence boundaries
            f.write('\n')

    print('Created file {}'.format(dataset_out_path))
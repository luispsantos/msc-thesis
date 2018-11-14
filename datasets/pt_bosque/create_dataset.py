import pandas as pd
from pathlib import Path
import yaml
import sys
import os

# importing from utils directory
sys.path.insert(1, os.path.join(sys.path[0], '..', 'utils'))
from rule_matcher import RuleMatcher
import contractions

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

    # read the input dataset in CoNNL-U format
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    df = pd.read_csv(dataset_in_path, sep='\t', names=column_names, skip_blank_lines=False, comment='#')

    # remove multi-word tokens by skipping all range IDs
    mwes = df[~(df.ID.isna() | df.ID.str.isdecimal())]['FORM']
    df = df[df.ID.isna() | df.ID.str.isdecimal()]

    # keep only a few columns of the dataset and discard the rest
    df = df[['FORM', 'UPOS']]
    df.rename(columns={'FORM': 'Token', 'UPOS': 'POS'}, inplace=True)

    # make contractions
    if keep_contractions:
        df = matcher.apply_rules(df)

    # define output format as joining columns with an output separator (e.g., ' ', '\t')
    # in a line-based format of a token per line where empty lines denote sentence boundaries
    output_rows = df.Token + output_separator + df.POS
    output_text = output_rows.str.cat(sep='\n', na_rep='')

    dataset_out_path.write_text(output_text)
    print('Created file {}'.format(dataset_out_path))

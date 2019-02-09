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
from util import *

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_separator, output_columns  = config['output_separator'], config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

rules, compound_words = rules['rules'], rules['compound_words']
hyphen_upos_map, token_upos_map = compound_words['hyphen_upos_map'], compound_words['token_upos_map']
token_upos_map = {token: upos_tag for upos_tag, tokens in token_upos_map.items() for token in tokens}

def create_exception_rules(exceptions):
    exception_rules = {}
    hyphen = {'Token': '-', 'UPOS': 'PUNCT'}
    concat_tokens = [{'Token': {'CONCAT': True}, 'UPOS': {'CONCAT': '+'}}]

    for exception_type, words in exceptions.items():
        for word in words:
            tokens = word.split('-')
            # create rule_in (e.g., [{'Token': 'sem'}, {'Token': '-', 'UPOS': 'PUNCT'}, {'Token': 'terra'}])
            rule_in = [{'Token': token} if idx % 2 == 0 else hyphen for token in tokens for idx in range(2)]
            rule_in = rule_in[:-1]  # remove last hyphen

            # create rule_out (either join together words into a single token or keep words separate)
            rule_out = concat_tokens if exception_type == 'single_words' else [{} for idx in range(len(rule_in))]

            exception_rules['+'.join(tokens)] = {'rule_in': rule_in, 'rule_out': rule_out}

    return exception_rules

matcher = RuleMatcher(rules)
matcher.add_rules(create_exception_rules(compound_words['exceptions']))

for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'pt_gsd-ud-{}.conllu'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    data_df = read_conllu(dataset_in_path)
    data_df = extract_multiwords(data_df)

    data_df, rule_counts = matcher.apply_rules(data_df)
    replace_values(hyphen_upos_map, data_df.UPOS)
    replace_values(token_upos_map, data_df.Token, data_df.UPOS)

    write_data(data_df, dataset_out_path, output_separator, output_columns)
    print('Created file {}'.format(dataset_out_path))


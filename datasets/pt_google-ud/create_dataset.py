from pathlib import Path
import pandas as pd
import sys
import os

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
compound_words, rules = rules['compound_words'], rules['rules']
hyphen_upos_map, token_upos_map = compound_words['hyphen_upos_map'], compound_words['token_upos_map']

token_upos_map = {token: upos_tag for upos_tag, tokens in token_upos_map.items() for token in tokens}

def create_exception_rules(exceptions):
    exception_rules = {}
    hyphen = {'Token': '-', 'UPOS': 'PUNCT'}
    concat_tokens = [{'Token': {'CONCAT': ''}, 'UPOS': {'CONCAT': '+'}}]

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

def process_dataset(data_in_path):
    # read CoNLL-U data and extract multi-word tokens
    data_df = read_conllu(data_in_path)
    data_df = extract_multiwords(data_df)

    # apply dataset-specific rules
    data_df, rule_counts = matcher.apply_rules(data_df)
    if 'train' in data_in_path.name:
        train_df = data_df
        train_counts = rule_counts
    globals().update(locals())

    # map compound words to a UPOS tag
    replace_values(hyphen_upos_map, data_df.UPOS)
    replace_values(token_upos_map, data_df.Token, data_df.UPOS)

    # map any remaining PART tags to ADJ
    data_df.UPOS.replace('PART', 'ADJ', inplace=True)

    return data_df

matcher = RuleMatcher(rules)
matcher.add_rules(create_exception_rules(compound_words['exceptions']))

# process dataset with pre-made data splits and write data to disk
data_in_format = 'pt_gsd-ud-{dataset_type}.conllu'.format
dataset_with_splits(process_dataset, data_in_format, dataset_in_dir, dataset_out_dir, output_columns)

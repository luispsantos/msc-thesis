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

def normalize_token(token_dict):
    return token_dict

for dataset_type in ['train']:
#for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'macmorpho-{}.txt'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    sents = []
    for raw_sent in dataset_in_path.open('r'):
        sent_tokens = []

        for raw_token in raw_sent.split():
            match = token_re.match(raw_token)

            #make sure the match included the whole token
            assert match, 'Token regex did not cover the whole token: {}'.format(raw_token)

            token_dict = match.groupdict()
            token_normalized = normalize_token(token_dict)

            sent_tokens.append(token_normalized)

        sents.append(sent_tokens)

    with dataset_out_path.open('w') as f:
        for sent_tokens in sents:
            for token_dict in sent_tokens:
                token_columns = [token_dict[col] for col in output_columns]

                token_line = output_separator.join(token_columns)
                f.write(token_line + '\n')

            # write an empty line to denote sentence boundaries
            f.write('\n')

        print('Created file {}'.format(dataset_out_path))

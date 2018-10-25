import pandas as pd
from pathlib import Path
import yaml

#read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_separator, output_columns  = config['output_separator'], config['output_columns']

# create output dataset directory if it doesn't exist
if not dataset_out_dir.exists():
    dataset_out_dir.mkdir(parents=True)

for dataset_type in ['train', 'dev', 'test']:
    dataset_in_path = dataset_in_dir / 'pt_bosque-ud-{}.conllu'.format(dataset_type)
    dataset_out_path = dataset_out_dir / '{}.txt'.format(dataset_type)

    # read the input dataset in CoNNL-U format
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    df = pd.read_csv(dataset_in_path, sep='\t', names=column_names, skip_blank_lines=False, comment='#')

    # remove multi-word tokens by skipping all range IDs
    df = df[df.ID.isna() | df.ID.str.isdecimal()]

    # keep only a few columns of the dataset and discard the rest
    df = df[['FORM', 'UPOS']]
    df.rename(columns={'FORM': 'Token', 'UPOS': 'POS'}, inplace=True)

    # output a line-based format similar to the CoNLL-2003 format
    with dataset_out_path.open('w') as f:
        for row in df.itertuples(index=False):
            # write an empty line to denote sentence boundaries
            if pd.isnull(row.Token):
                f.write('\n')
            else:
                f.write(row.Token + output_separator + row.POS + '\n')

        print('Created file {}'.format(dataset_out_path))

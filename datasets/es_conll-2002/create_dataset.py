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
pos_tagset_ud_map = rules['pos_tagset_ud_map']

conll_files = {'train': 'esp.train', 'dev': 'esp.testa', 'test': 'esp.testb'}

for dataset_type, conll_file in conll_files.items():
    data_in_path = dataset_in_dir / conll_file
    data_out_path = dataset_out_dir / (dataset_type + '.txt')

    # read CoNLL-2002 data
    data_df = read_conll(data_in_path, sep=' ', column_names=['Token', 'POS', 'NER'])

    # obtain the first and first+second characters of the POS tag (e.g. VMIS3S0 -> V and VM)
    pos = data_df.POS.astype('category')
    data_df['POS_0'], data_df['POS_0+1'] = pos.str[:1], pos.str[:2]

    # map POS tagset to UD tagset based on the first and first+second characters of POS tags
    data_df['UPOS'] = data_df['POS_0'].map(pos_tagset_ud_map)
    replace_values(pos_tagset_ud_map, data_df['POS_0+1'], data_df.UPOS)

    # write data to disk
    write_data(data_df, data_out_path, output_columns)


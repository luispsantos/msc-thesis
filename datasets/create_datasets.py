from pathlib import Path
from os.path import join
from itertools import chain
from zipfile import ZipFile
import subprocess
import yaml

class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

def process_config(config):
    """Selectively choose the info on config.yml to retain."""
    # remove keys of dataset input/output directories
    del config['dataset_in_dir']
    del config['dataset_out_dir']

    # rename output_columns key to columns
    config['columns'] = config.pop('output_columns')

    return config

# open the datasets zip file
datasets = ZipFile('datasets.zip', 'w')
cwd = Path.cwd()

# iterate over the Portuguese and Spanish datasets
for dataset_dir in chain(cwd.glob('pt_*'), cwd.glob('es_*')):
    dataset_name = dataset_dir.name
    data_dir = dataset_dir / 'data'

    # create dataset if it has not yet been created
    if not data_dir.exists():
        print(f'Creating dataset {dataset_name}')
        subprocess.run(['python', 'create_dataset.py'], cwd=dataset_dir)

    # copy data files to the datasets zip file
    for data_file in data_dir.iterdir():
        datasets.write(data_file, join(dataset_name, data_file.name))

    # copy YAML file with dataset statistics
    stats_file = dataset_dir / 'stats.yml'
    datasets.write(stats_file, join(dataset_name, 'stats.yml'))

    # load the dataset's configuration file
    with open(dataset_dir / 'config.yml') as f:
        config = yaml.safe_load(f)

    # write the information inside the configuration file
    config_str = yaml.dump(process_config(config), Dumper=MyDumper, sort_keys=False)
    datasets.writestr(join(dataset_name, 'config.yml'), config_str)

# copy datasets to the EMNLP BiLSTM-CNN-CRF data folder
destination_dir = Path('../emnlp2017-bilstm-cnn-crf/data/')
datasets.extractall(destination_dir)

print('Created zip file datasets.zip')
datasets.close()

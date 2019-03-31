from pathlib import Path
from os.path import join, getmtime
from itertools import chain
from zipfile import ZipFile
import subprocess
import yaml

def process_config(config):
    # remove keys of dataset directories
    del config['dataset_in_dir']
    del config['dataset_out_dir']

    # rename output_columns to columns
    config['columns'] = config.pop('output_columns')

    return config

# obtain last modification time of zip file
zipfile_path = Path('datasets.zip')
zipfile_mtime = getmtime(zipfile_path) if zipfile_path.exists() else 0.0

# open the datasets zip file
datasets = ZipFile(zipfile_path, 'w')
cwd = Path.cwd()

# iterate over the Portuguese and Spanish datasets
for dataset_dir in chain(cwd.glob('pt_*'), cwd.glob('es_*')):
    dataset_name = dataset_dir.name
    data_dir = dataset_dir / 'data'

    # create dataset if it has not been created or
    # if it has been modified later than the zip file
    if not data_dir.exists() or getmtime(data_dir) > zipfile_mtime:
        print(f'Creating dataset {dataset_name}')
        subprocess.run(['python', 'create_dataset.py'], cwd=dataset_dir)

    # copy data files to the datasets zip file
    for data_file in data_dir.iterdir():
        datasets.write(data_file, join(dataset_name, data_file.name))

    # load the dataset's configuration file
    with open(dataset_dir / 'config.yml') as f:
        config = yaml.safe_load(f)

    # write the information inside the configuration file
    config_str = yaml.dump(process_config(config), sort_keys=False)
    datasets.writestr(join(dataset_name, 'config.yml'), config_str)

# copy datasets to the EMNLP BiLSTM-CNN-CRF data folder
destination_dir = Path('../emnlp2017-bilstm-cnn-crf/data/')
datasets.extractall(destination_dir)

print('Created zip file datasets.zip')
datasets.close()

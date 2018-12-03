import pandas as pd
import csv
from pathlib import Path
import yaml

class InvalidDatasetError(Exception):
    """Raise when a dataset cannot be successfully loaded"""

class Dataset:

    def __init__(self, dataset_path):
        self.load(dataset_path)

    def __iter__(self):
        for dataset_type in ['train', 'dev', 'test']:
            yield dataset_type, self.data[dataset_type]

    def load(self, dataset_path):
        if not dataset_path.exists():
            raise InvalidDatasetError('Dataset {} does not exist'.format(dataset_path))

        try:
            config_file = dataset_path / 'config.yml'
            with config_file.open('r') as f:
                self.config = yaml.load(f)
        except Exception:
            raise InvalidDatasetError('Cannot open configuration file {}'.format(config_file))
            
        data_dir = dataset_path / self.config['dataset_out_dir']
        if not data_dir.exists():
             raise InvalidDatasetError('Dataset {} is missing a {} folder (run create_dataset.py)'.format(dataset_dir, data_dir.name))

        separator, column_names = self.config['output_separator'], self.config['output_columns']
        self.data = {}

        for dataset_type in ['train', 'dev', 'test']:
            data_file = data_dir / '{}.txt'.format(dataset_type)
            if not data_file.exists():
                raise InvalidDatasetError('Missing data file {}'.format(data_file))

            self.data[dataset_type] = pd.read_csv(data_file, sep=separator, names=column_names, skip_blank_lines=False, quoting=csv.QUOTE_NONE)


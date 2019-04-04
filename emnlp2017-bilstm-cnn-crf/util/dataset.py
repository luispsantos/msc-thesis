from pathlib import Path
import yaml

class InvalidDatasetError(Exception):
    """Raise when a dataset cannot be successfully loaded"""

class Dataset:
    # map dataset column names to names expected by the application
    column_map = {'Token': 'tokens', 'UPOS': 'POS', 'NER': 'NER_BIO'}

    def __init__(self, dataset_name, evaluate=True):
        self.dataset_name = dataset_name
        self.evaluate = evaluate
        self.validate(dataset_name)

        columns = self.config['columns']
        columns = [self.map_column_name(col_name) for col_name in columns]

        self.columns = {idx: col for idx, col in enumerate(columns)}
        self.predict_labels = [col for col in columns if col in ('POS', 'NER_BIO')]

    def validate(self, dataset_name):
        dataset_path = Path('data', dataset_name)
        if not dataset_path.exists():
            raise InvalidDatasetError(f'Dataset {dataset_name} does not exist at {dataset_path}')

        for dataset_type in ['train', 'dev', 'test']:
            data_file = dataset_path / (dataset_type + '.txt')
            if not data_file.exists():
                raise InvalidDatasetError(f'Missing data file {data_file}')

        try:
            config_file = dataset_path / 'config.yml'
            with config_file.open('r') as f:
                self.config = yaml.safe_load(f)
        except Exception:
            raise InvalidDatasetError(f'Cannot open configuration file {config_file}')

    def map_column_name(self, col_name):
        return self.column_map.get(col_name, col_name)

    def to_dict(self, predict_label):
        if predict_label not in self.predict_labels:
            raise InvalidDatasetError(f'Cannot use label {predict_label} for prediction')

        dataset_params = {
            'columns': self.columns,
            'label': predict_label,
            'evaluate': self.evaluate,
            'commentSymbol': None
        }
        dataset = {self.dataset_name: dataset_params}

        return dataset

def iter_datasets(lang, task):
    """Iterates along datasets for a specific language and task."""
    data_dir = Path('data')
    # iterate through the datasets for the target language
    for dataset_dir in sorted(data_dir.glob(f'{lang}_*')):
        dataset_name = dataset_dir.name
        dataset = Dataset(dataset_name)

        # check if this dataset contains the task of interest
        predict_label = dataset.map_column_name(task)
        if predict_label in dataset.predict_labels:
            yield dataset_name, dataset.to_dict(predict_label)


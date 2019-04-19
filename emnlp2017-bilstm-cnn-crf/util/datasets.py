from pathlib import Path
import yaml

class InvalidDatasetError(Exception):
    """Raise when a dataset cannot be successfully loaded"""

class Dataset:
    # map dataset column names to names expected by the application
    column_map = {'Token': 'tokens', 'UPOS': 'POS', 'NER': 'NER_BIO'}

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.validate(dataset_id)

        lang = dataset_id.split('_')[0]
        self.lang = lang.upper()

        columns = self.config['columns']
        self.name = self.config['name']

        columns = [self.column_map.get(col, col) for col in columns]
        self.columns = {idx: col for idx, col in enumerate(columns)}

        self.labels = [col for col in columns if col in ('POS', 'NER_BIO')]
        self.tasks = [col.replace('_BIO', '') for col in self.labels]

    def validate(self, dataset_id):
        dataset_path = Datasets.datasets_dir / dataset_id
        if not dataset_path.exists():
            raise InvalidDatasetError(f'Dataset {dataset_id} does not exist at {dataset_path}')

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

    def __iter__(self):
        for task in self.tasks:
            yield task, self.to_dict(task)

    def to_dict(self, task=None):
        label = task + '_BIO' if task == 'NER' else task
        if label is None:
            predict_label = self.labels
        elif label in self.labels:
            predict_label = [label]
        else:
            raise InvalidDatasetError(f'Cannot use label {label} for prediction')

        dataset_params = {
            'columns': self.columns,
            'label': predict_label,
            'evaluate': True,
            'commentSymbol': None
        }
        dataset = {self.dataset_id: dataset_params}

        return dataset

class Datasets:
    datasets_dir = Path('data')

    def __init__(self, names=None, exclude=None, lang=None, task=None):
        self.datasets = {}
        self.lang, self.task = lang, task

        # iterate through the multiple datasets
        for dataset_path in sorted(self.datasets_dir.iterdir()):
            dataset_id = dataset_path.name
            dataset = Dataset(dataset_id)

            # selectively choose the datasets
            if (names is None or dataset_id in names) \
            and (exclude is None or dataset_id not in exclude) \
            and (lang is None or lang == dataset.lang) \
            and (task is None or task in dataset.tasks):
                self.datasets[dataset_id] = dataset

    def __iter__(self):
        yield from self.datasets.items()

    def to_dict(self):
        datasets = {}
        for dataset in self.datasets.values():
            dataset_dict = dataset.to_dict(self.task)
            datasets.update(dataset_dict)

        return datasets

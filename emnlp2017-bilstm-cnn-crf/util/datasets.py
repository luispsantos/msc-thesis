from pathlib import Path
import yaml

class InvalidDatasetError(Exception):
    """Raise when a dataset cannot be successfully loaded"""

class Dataset:
    # map dataset column names to names expected by the application
    column_map = {'Token': 'tokens', 'UPOS': 'POS', 'NER': 'NER_BIO'}

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.lang = dataset_name.split('_')[0]
        self.validate(dataset_name)

        columns = self.config['columns']
        columns = [self.column_map.get(col, col) for col in columns]

        self.columns = {idx: col for idx, col in enumerate(columns)}
        self.tasks = [col for col in columns if col in ('POS', 'NER_BIO')]

    def validate(self, dataset_name):
        dataset_path = Datasets.datasets_dir / dataset_name
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

    def to_dict(self, task=None):
        if task is None:
            tasks = self.tasks
        elif task not in self.columns.values():
            raise InvalidDatasetError(f'Cannot use label {task} for prediction')
        else:
            tasks = [task]

        dataset_params = {
            'columns': self.columns,
            'label': tasks,
            'evaluate': True,
            'commentSymbol': None
        }
        dataset = {self.dataset_name: dataset_params}

        return dataset

class Datasets:
    datasets_dir = Path('data')

    def __init__(self, names=None, exclude=None, lang=None, task=None):
        self.datasets = {}
        task = task + '_BIO' if task == 'NER' else task
        self.lang, self.task = lang, task

        # iterate through the multiple datasets
        for dataset_path in sorted(self.datasets_dir.iterdir()):
            dataset_name = dataset_path.name
            dataset = Dataset(dataset_name)

            # selectively choose the datasets
            if (names is None or dataset_name in names) \
            and (exclude is None or dataset_name not in exclude) \
            and (lang is None or lang == dataset.lang)  \
            and (task is None or task in dataset.tasks):
                self.datasets[dataset_name] = dataset

    def __iter__(self):
        return self.datasets.items()

    def to_dict(self):
        datasets = {}
        for dataset in self.datasets.values():
            dataset_dict = dataset.to_dict(self.task)
            datasets.update(dataset_dict)

        return datasets

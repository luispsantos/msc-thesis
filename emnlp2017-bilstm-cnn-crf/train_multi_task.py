import os
import logging
import sys
from pathlib import Path
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle
from util.datasets import Datasets

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# paths for input/output directories
embeddings_dir = Path('embeddings')
models_dir, results_dir = Path('models/multi_task'), Path('results/multi_task')

# select network hyperparameters
network_params = {'charEmbeddings': 'CNN', 'LSTM-Size': [100, 100], 'classifier': 'CRF',
                  'dropout': (0.5, 0.5), 'earlyStopping': 15, 'miniBatchSize': 32}

# number of runs with different random initialization
max_runs = 1

def run_experiment(datasets, lang, task, embeddings_path, max_runs):
    # prepares the datasets to be used with the LSTM-network
    # creates and stores pickle files in the pkl/ folder
    pickleFile = prepareDataset(embeddings_path, datasets, None, True)

    # loads the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    lang_prefix = f'{lang.lower()}_' if lang is not None else ''
    task_suffix = f'_{task.lower()}' if task is not None else ''
    experiment_name = lang_prefix + 'datasets' + task_suffix

    # run the model multiple times with the same hyperparameters and different random initialization
    for run_idx in range(max_runs): 
        # set network hyperparameters and mappings/datasets
        model = BiLSTM(network_params)
        model.setMappings(mappings, embeddings)
        model.setDataset(datasets, data)

        # path to store the trained model and model results
        model.modelSavePath = models_dir / f'{experiment_name}.h5'
        model.storeResults(results_dir / f'{experiment_name}.csv')

        # build and train the model
        model.buildModel()
        model.fit(epochs=500)  # do not limit training by epochs - use early stopping


multitask_datasets = [
    Datasets(exclude=['pt_colonia', 'pt_wikiner'], lang='PT', task='POS'),
    Datasets(exclude=['pt_wikiner'], lang='PT', task='NER'),
    Datasets(exclude=['pt_colonia', 'pt_wikiner'], lang='PT')
]

# iterate through the multiple language and task dataset combinations
for datasets in multitask_datasets:
    # obtain datasets' language and task
    lang, task = datasets.lang, datasets.task

    # select fasttext word embeddings
    embeddings_path = embeddings_dir / f'{lang.lower()}.fasttext.oov.vec.gz'

    pid = os.fork()
    if pid == 0:  # child process
        # run experiment in a separate process so that memory is released at the end
        run_experiment(datasets.to_dict(), lang, task, embeddings_path, max_runs)
        sys.exit(0)
    else:  # parent process
        pid, status = os.waitpid(pid, 0)
        assert status == 0, f'Child process returned with failed status code: {status}'
        logger.info(f'Completed experiment: lang {"all" if lang is None else lang} - ' \
                                           f'task {"all" if task is None else task}')


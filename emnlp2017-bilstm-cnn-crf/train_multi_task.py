import os
import logging
import sys
from pathlib import Path
from multiprocessing import Process
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import prepareDatasets, loadDatasetPickle
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

# select BiLSTM network hyperparameters
network_params = {'charEmbeddings': 'LSTM', 'charLSTMSize': 25, 'LSTM-Size': [100, 100],
                  'classifier': 'CRF', 'optimizer': 'adam' ,'dropout': (0.5, 0.5),
                  'adversarial': False, 'earlyStopping': 10}

def run_experiment(datasets_dict, lang, task, embeddings, mappings, data):
    # set network hyperparameters and mappings/datasets
    model = BiLSTM(network_params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets_dict, data)

    # define the experiment name
    lang_prefix = f'{lang.lower()}_' if lang is not None else ''
    task_suffix = f'_{task.lower()}' if task is not None else ''
    experiment_name = lang_prefix + 'datasets' + task_suffix

    # path to store the trained model and model results
    model.modelSavePath = models_dir / f'{experiment_name}.h5'
    model.storeResults(results_dir / f'{experiment_name}.csv')

    # build and train the model
    model.buildModel()
    model.fit(epochs=500)  # do not limit training by epochs - use early stopping

for lang in ['PT', 'ES', None]:
    # select fasttext word embeddings
    lang_prefix = lang.lower() if lang is not None else 'es2pt'
    embeddings_path = embeddings_dir / f'{lang_prefix}.fasttext.oov.vec.gz'

    # prepare the datasets to be used with the LSTM network
    prepareDatasets(embeddings_path, lang)

    # load the embeddings and the datasets
    embeddings, mappings, data = loadDatasetPickle(embeddings_path, lang)

    # iterate through the multiple dataset combinations of language and task
    for task in ['POS', 'NER', None]:
        if lang is None and task is not None: continue
        # obtain datasets for the experiment
        datasets = Datasets(exclude=['pt_colonia'], lang=lang, task=task)
        datasets_dict = datasets.to_dict()

        # run experiment in a separate process so that memory is released at the end
        proc_args = (datasets_dict, lang, task, embeddings, mappings, data)
        proc = Process(target=run_experiment, args=proc_args)

        proc.start(); proc.join()
        logger.info(f'Completed experiment: lang {"all" if lang is None else lang} - ' \
                                           f'task {"all" if task is None else task}')


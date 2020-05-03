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
embeddings_dir, multi_task_models_dir = Path('embeddings'), Path('models/multi_task')
models_dir, results_dir = Path('models'), Path('results')

def run_experiment(dataset_id, dataset_dict, lang, task, data):
    # load the pre-trained BiLSTM model
    lang_prefix = f'{lang.lower()}_' if lang is not None else ''
    model = BiLSTM.loadModel(multi_task_models_dir / f'{lang_prefix}datasets.h5')

    # set the single task dataset and select both tasks
    model.setDataset(dataset_dict, data)
    model.tasks = ['POS', 'NER_BIO']

    # path to store the trained model and model results
    experiment_name = f'{dataset_id}_{task.lower()}'
    pretrain_type = 'multi_task' if lang is not None else 'cross_lingual'

    model.modelSavePath = models_dir / f'pretrain_{pretrain_type}/{experiment_name}.h5'
    model.storeResults(results_dir / f'pretrain_{pretrain_type}/{experiment_name}.csv')

    # train the model - no need to build model here
    model.fit(epochs=500)  # do not limit training by epochs - use early stopping

for lang in ['PT', 'ES', None]:
    # select fasttext word embeddings
    lang_prefix = lang.lower() if lang is not None else 'es2pt'
    embeddings_path = embeddings_dir / f'{lang_prefix}.fasttext.oov.vec.gz'

    # prepare the datasets to be used with the LSTM network
    prepareDatasets(embeddings_path, lang)

    # load the embeddings and the datasets
    embeddings, mappings, data = loadDatasetPickle(embeddings_path, lang)

    # obtain datasets for the experiment
    datasets = Datasets(exclude=['pt_colonia'], lang=lang)

    # iterate through the multiple Portuguese and Spanish datasets
    for dataset_id, dataset in datasets:

        # iterate through the tasks (POS and NER) for each dataset
        for task, dataset_dict in dataset:
            # run experiment in a separate process so that memory is released at the end
            proc_args = (dataset_id, dataset_dict, lang, task, data)
            proc = Process(target=run_experiment, args=proc_args)

            proc.start(); proc.join()
            logger.info(f'Completed experiment: {dataset_id} - {task}')


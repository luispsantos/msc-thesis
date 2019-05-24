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
models_dir, results_dir = Path('models/single_task'), Path('results/single_task')

# select BiLSTM network hyperparameters
network_params = {'charEmbeddings': 'LSTM', 'charLSTMSize': 25, 'LSTM-Size': [100, 100],
                  'classifier': 'CRF','dropout': (0.5, 0.5), 'earlyStopping': 10}

def run_experiment(dataset_id, dataset_dict, task, embeddings, mappings, data):
    # set network hyperparameters and mappings/datasets
    model = BiLSTM(network_params)
    model.setMappings(mappings, embeddings)
    model.setDataset(dataset_dict, data)

    # path to store the trained model and model results
    experiment_name = f'{dataset_id}_{task.lower()}'
    model.modelSavePath = models_dir / f'{experiment_name}.h5'
    model.storeResults(results_dir / f'{experiment_name}.csv')

    # build and train the model
    model.buildModel()
    model.fit(epochs=500)  # do not limit training by epochs - use early stopping

for lang in ['PT', 'ES']:
    # select fasttext word embeddings
    embeddings_path = embeddings_dir / f'{lang.lower()}.fasttext.oov.vec.gz'

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
            proc_args = (dataset_id, dataset_dict, task, embeddings, mappings, data)
            proc = Process(target=run_experiment, args=proc_args)

            proc.start(); proc.join()
            logger.info(f'Completed experiment: {dataset_id} - {task}')


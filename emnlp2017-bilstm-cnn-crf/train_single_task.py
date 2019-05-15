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
models_dir, results_dir = Path('models/single_task'), Path('results/single_task')

# BiLSTM network hyperparameters
network_params = {'charEmbeddings': 'LSTM', 'charLSTMSize': 25, 'LSTM-Size': [100, 100],
                  'classifier': 'CRF','dropout': (0.5, 0.5), 'earlyStopping': 10}

# number of runs with different random initialization
max_runs = 1

def run_experiment(dataset_id, dataset_dict, task, embeddings_path, max_runs):
    # prepares the dataset to be used with the LSTM-network
    # creates and stores pickle files in the pkl/ folder
    pickleFile = prepareDataset(embeddings_path, dataset_dict, None, True)

    # loads the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    experiment_name = f'{dataset_id}_{task.lower()}'

    # run the model multiple times with the same hyperparameters and different random initialization
    for run_idx in range(max_runs): 
        # set network hyperparameters and mappings/datasets
        model = BiLSTM(network_params)
        model.setMappings(mappings, embeddings)
        model.setDataset(dataset_dict, data)

        # path to store the trained model and model results
        model.modelSavePath = models_dir / f'{experiment_name}.h5'
        model.storeResults(results_dir / f'{experiment_name}.csv')

        # build and train the model
        model.buildModel()
        model.fit(epochs=500)  # do not limit training by epochs - use early stopping

datasets = Datasets()

# iterate through the multiple Portuguese and Spanish datasets
for dataset_id, dataset in datasets:
    # obtain the dataset's language
    lang = dataset.lang

    # select fasttext word embeddings
    embeddings_path = embeddings_dir / f'{lang.lower()}.fasttext.oov.vec.gz'

    # iterate through the tasks (POS and NER) for a dataset
    for task, dataset_dict in dataset:
        # run experiment in a separate process so that memory is released at the end
        pid = os.fork()
        if pid == 0:  # child process
            run_experiment(dataset_id, dataset_dict, task, embeddings_path, max_runs)
            sys.exit(0)
        else:  # parent process
            pid, status = os.waitpid(pid, 0)
            assert status == 0, f'Child process returned with failed status code: {status}'
            logger.info(f'Completed experiment: {dataset_id} - {task}')


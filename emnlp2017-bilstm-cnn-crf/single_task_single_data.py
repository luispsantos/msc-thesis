import os
import logging
import sys
import csv
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import prepareDataset, loadDatasetPickle
from util.dataset import iter_datasets

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

# select network hyperparameters
network_params = {'charEmbeddings': 'CNN', 'LSTM-Size': [100, 100], 'classifier': ['CRF'],
                  'dropout': (0.5, 0.5), 'earlyStopping': 5, 'miniBatchSize': 32}

# number of runs with different random initialization
max_runs = 1

def run_experiment(csv_file, writer, dataset_name, dataset, task, embeddings, max_runs):
    # prepares the dataset to be used with the LSTM-network
    # creates and stores pickle files in the pkl/ folder
    pickleFile = prepareDataset(embeddings, dataset, None, True)

    # load the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    # run the model multiple times with the same hyperparameters and different random initialization
    for run_idx in range(max_runs): 
        # build and train the model
        model = BiLSTM(network_params)
        model.setMappings(mappings, embeddings)
        model.setDataset(dataset, data)

        # specify path to store the model and model results
        model.modelSavePath = f'models/{dataset_name}_{task}.h5'
        model.storeResults(f'results/{dataset_name}_{task}.csv')

        model.buildModel()
        model.fit(epochs=500)  # do not limit training by epochs - use early stopping

        writer.writerow([dataset_name, task, run_idx+1,
                         model.max_dev_score[dataset_name], model.max_test_score[dataset_name]])
        csv_file.flush()

with open('results/single_task_single_data.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['dataset', 'task', 'run', 'dev_score', 'test_score'])
    csv_file.flush()

    for lang in ['pt', 'es']:
        # select fasttext word embeddings
        embeddings = f'embeddings/cc.{lang}.300.vec.gz'

        # iterate through the POS and NER datasets for each language
        for task in ['POS', 'NER']:
            for dataset_name, dataset in iter_datasets(lang, task):
                pid = os.fork()
                if pid == 0:  # child process
                    run_experiment(csv_file, writer, dataset_name, dataset, task, embeddings, max_runs)
                    sys.exit(0)
                else:  # parent process
                    pid, status = os.waitpid(pid, 0)
                    assert status == 0, f'Child process returned with failed status code: {status}'
                    logger.info(f'Completed experiment: {dataset_name} - {task}')


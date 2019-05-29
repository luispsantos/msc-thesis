from pathlib import Path
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import loadDatasetPickle
from util.datasets import Dataset
from evaluator import Evaluator

# paths for input/output directories
models_dir, embeddings_dir = Path('models/single_task'), Path('embeddings')
pkl_dir, tables_dir = Path('pkl'), Path('tables')

BaseManager.register('Evaluator', Evaluator)
manager = BaseManager(); manager.start()

loaded_datasets = {}
evaluator = manager.Evaluator()

def eval_single_task(model_path, dataset_id, task, evaluator, embeddings, mappings, data):
    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)

    # create dataset dictionary
    dataset = Dataset(dataset_id)
    dataset_dict = dataset.to_dict(task)

    # set the model mappings and datasets
    model.setMappings(mappings, embeddings)
    model.setDataset(dataset_dict, data)

    # obtain mapping of indices to POS/NER labels
    label = task + '_BIO' if task == 'NER' else task
    idx2label = model.idx2Labels[label]

    # obtain train and test data
    train_data = data[dataset_id]['trainMatrix']
    test_data = data[dataset_id]['testMatrix']

    # obtain correct and predicted sentences
    corr_idxs = [sentence[label] for sentence in test_data]
    pred_idxs = model.predictLabels(test_data)[label]

    # convert indices to labels (POS tags or NER tags in BIO format)
    corr_labels = [[idx2label[idx] for idx in sent] for sent in corr_idxs]
    pred_labels = [[idx2label[idx] for idx in sent] for sent in pred_idxs]

    evaluator.eval(dataset.name, dataset.lang, task, corr_labels,
                   pred_labels, train_data, test_data)
    print(f'Evaluated single_task - {dataset_id} - {task}')

# iterate through the saved models
for model_path in sorted(models_dir.glob('*.h5')):
    model_name = model_path.stem

    # obtain dataset ID and task from model name
    dataset_id, task = model_name.rsplit('_', 1)
    task = task.upper()

    # obtain dataset language from dataset ID
    lang = dataset_id.split('_')[0]
    lang = lang.upper()

    if lang not in loaded_datasets:
        # select fasttext word embeddings
        embeddings_path = embeddings_dir / f'{lang.lower()}.fasttext.oov.vec.gz'

        # load and cache the embeddings, mappings and datasets
        loaded_datasets[lang] = loadDatasetPickle(embeddings_path, lang)

    # unpack the embeddings, mappings and datasets
    embeddings, mappings, data = loaded_datasets[lang]

    # evaluate model in a separate process so that memory is released at the end
    proc_args = (model_path, dataset_id, task, evaluator, embeddings, mappings, data)
    proc = Process(target=eval_single_task, args=proc_args)

    proc.start(); proc.join()

# write the evaluation tables
evaluator.write_tables(tables_dir / 'single_task')

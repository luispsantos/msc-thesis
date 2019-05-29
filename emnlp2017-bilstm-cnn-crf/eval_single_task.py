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
manager = BaseManager()

manager.start()
evaluator = manager.Evaluator()

def eval_single_task(model_path, evaluator, embeddings, mappings, data):
    # obtain dataset ID and task
    dataset_id, task = model_path.stem.rsplit('_', 1)
    task = task.upper()

    # obtain dataset name and language
    dataset = Dataset(dataset_id)
    dataset_name, lang = dataset.name, dataset.lang

    # obtain train and test data
    train_data = data[dataset_id]['trainMatrix']
    test_data = data[dataset_id]['testMatrix']

    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)
    model.setMappings(mappings, embeddings)

    # set the model dataset
    dataset_dict = dataset.to_dict(task)
    model.setDataset(dataset_dict, data)

    # obtain mapping of indices to POS/NER labels
    label = task + '_BIO' if task == 'NER' else task
    idx2label = model.idx2Labels[label]

    # obtain correct and predicted sentences
    corr_idxs = [sentence[label] for sentence in test_data]
    pred_idxs = model.predictLabels(test_data)[label]

    # convert indices to labels (POS tags or NER tags in BIO format)
    corr_labels = [[idx2label[idx] for idx in sent] for sent in corr_idxs]
    pred_labels = [[idx2label[idx] for idx in sent] for sent in pred_idxs]

    evaluator.eval(dataset_name, lang, task, corr_labels, pred_labels, train_data, test_data)
    print(f'Evaluated single_task - {dataset_id} - {task}')

for lang in ['PT', 'ES']:
    # select fasttext word embeddings
    embeddings_path = embeddings_dir / f'{lang.lower()}.fasttext.oov.vec.gz'

    # load the embeddings and the datasets
    embeddings, mappings, data = loadDatasetPickle(embeddings_path, lang)

    # iterate through the saved models
    for model_path in sorted(models_dir.glob(f'{lang.lower()}_*.h5')):
        # evaluate model in a separate process so that memory is released at the end
        proc_args = (model_path, evaluator, embeddings, mappings, data)
        proc = Process(target=eval_single_task, args=proc_args)

        proc.start(); proc.join()

# write the evaluation tables
evaluator.write_tables(tables_dir / 'single_task')

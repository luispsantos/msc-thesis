from pathlib import Path
from itertools import chain
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import loadDatasetPickle
from util.datasets import Datasets
from evaluator import Evaluator

# paths for input/output directories
models_dir, embeddings_dir = Path('models/multi_task'), Path('embeddings')
pkl_dir, tables_dir = Path('pkl'), Path('tables')

BaseManager.register('Evaluator', Evaluator)
manager = BaseManager(); manager.start()

loaded_datasets = {}
evaluators = {transfer_setting: manager.Evaluator() for transfer_setting
              in ['out_of_domain', 'cross_domain', 'multi_task', 'cross_lingual']}

def eval_multi_task(model_path, lang, task, evaluators, embeddings, mappings, data):
    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)
    print(f'Loaded model {model_path}')

    # obtain the evaluator based on the transfer setting
    if model_path.parent.name == 'single_task':
        transfer_setting = 'out_of_domain'
    elif lang is not None and task is not None:
        transfer_setting = 'cross_domain'
    elif lang is not None and task is None:
        transfer_setting = 'multi_task'
    elif lang is None and task is None:
        transfer_setting = 'cross_lingual'
    else:
        raise ValueError('Unknown transfer setting')

    evaluator = evaluators[transfer_setting]

    # create datasets dictionary
    datasets = Datasets(lang=lang, task=task)
    datasets_dict = datasets.to_dict()

    # set the model mappings and datasets
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets_dict, data)

    # evaluate each dataset separately
    for dataset_id, dataset in datasets:
        # obtain train and test data
        train_data = data[dataset_id]['trainMatrix']
        test_data = data[dataset_id]['testMatrix']

        # predict labels for the POS and NER tasks
        task_predictions = model.predictLabels(test_data)

        # iterate through the available output tasks
        for label in model.labelKeys[dataset_id]:
            # obtain mapping of indices to POS/NER labels
            task = label.replace('_BIO', '')
            idx2label = model.idx2Labels[label]

            # obtain correct and predicted sentences
            corr_idxs = [sentence[label] for sentence in test_data]
            pred_idxs = task_predictions[label]

            # convert indices to labels (POS tags or NER tags in BIO format)
            corr_labels = [[idx2label[idx] for idx in sent] for sent in corr_idxs]
            pred_labels = [[idx2label[idx] for idx in sent] for sent in pred_idxs]

            evaluator.eval(dataset.name, dataset.lang, task, corr_labels,
                           pred_labels, train_data, test_data)
            print(f'Evaluated {transfer_setting} - {dataset_id} - {task}')

# evaluate CINTIL and CoNLL-02 on the remaining datasets
single_models_dir = Path('models/single_task')
single_models = [single_models_dir.glob(model_name + '_*.h5')
                         for model_name in ('pt_cintil', 'es_conll-2002')]

# iterate through the saved models
for model_path in sorted(chain(models_dir.glob('*.h5'), *single_models)):
    model_name = model_path.stem

    # extract language and task from model name
    model_parts = model_name.split('_')
    lang, task = model_parts[0], model_parts[-1]

    # obtain the model's language and task
    lang = lang.upper() if lang in ('pt', 'es') else None
    task = task.upper() if task in ('pos', 'ner') else None

    if lang not in loaded_datasets:
        # select fasttext word embeddings
        lang_prefix = lang.lower() if lang is not None else 'es2pt'
        embeddings_path = embeddings_dir / f'{lang_prefix}.fasttext.oov.vec.gz'

        # load and cache the embeddings, mappings and datasets
        loaded_datasets[lang] = loadDatasetPickle(embeddings_path, lang)

    # unpack the embeddings, mappings and datasets
    embeddings, mappings, data = loaded_datasets[lang]

    # evaluate model in a separate process so that memory is released at the end
    proc_args = (model_path, lang, task, evaluators, embeddings, mappings, data)
    proc = Process(target=eval_multi_task, args=proc_args)

    proc.start(); proc.join()

# write the evaluation tables
for transfer_setting, evaluator in evaluators.items():
    evaluator.write_tables(tables_dir / transfer_setting)

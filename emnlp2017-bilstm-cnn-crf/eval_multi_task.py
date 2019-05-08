from pathlib import Path
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import loadDatasetPickle
from util.datasets import Datasets
from evaluator import Evaluator

# paths for input/output directories
models_dir, results_dir = Path('models/multi_task'), Path('results/multi_task')
pkl_dir, tables_dir = Path('pkl'), Path('tables')

evaluators = {transfer_setting: Evaluator() for transfer_setting
                                            in ['cross_domain', 'multi_task']}

for model_path in sorted(models_dir.glob('*.h5')):
    model_name = model_path.stem

    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)
    print(f'Loading model {model_name}')

    # extract language and task from model name
    model_parts = model_name.split('_')
    lang, task = model_parts[0], model_parts[-1]

    # obtain the model's language and task
    lang = lang.upper() if lang in ('pt', 'es') else None
    task = task.upper() if task in ('pos', 'ner') else None

    # obtain the evaluator based on the transfer setting
    if lang is not None and task is not None:
        transfer_setting = 'cross_domain'
    elif lang is not None and task is None:
        transfer_setting = 'multi_task'

    evaluator = evaluators[transfer_setting]

    # obtain the IDs of all datasets used in training
    dataset_ids = model.labelKeys.keys()
    dataset_ids_str = '_'.join(dataset_ids)

    # filename for fasttext word embeddings
    if lang is not None:
        embeddings_name = f'{lang.lower()}.fasttext.oov.vec'

    # load the datasets
    pkl_file = pkl_dir / f'{dataset_ids_str}_{embeddings_name}.pkl'
    embeddings, mappings, data = loadDatasetPickle(pkl_file)

    # create datasets dictionary
    datasets = Datasets(names=dataset_ids, lang=lang, task=task)
    datasets_dict = datasets.to_dict()

    # set the model mappings and dataset
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets_dict, data)

    # evaluate each dataset separately
    for dataset_id, dataset in datasets:
        # obtain train and test data
        train_data = data[dataset_id]['trainMatrix']
        test_data = data[dataset_id]['testMatrix']

        # predict labels for POS and NER tasks
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

            evaluator.eval(dataset.name, lang, task, corr_labels, pred_labels, train_data, test_data)
            print(f'Evaluated {transfer_setting} - {dataset_id} - {task}')

# write the evaluation tables
for transfer_setting, evaluator in evaluators.items():
    evaluator.write_tables(tables_dir / transfer_setting)

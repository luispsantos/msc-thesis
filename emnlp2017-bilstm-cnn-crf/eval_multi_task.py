from pathlib import Path
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import loadDatasetPickle
from util.datasets import Datasets
from evaluate import Evaluate

# paths for input/output directories
models_dir, results_dir = Path('models/multi_task'), Path('results/multi_task')
pkl_dir, tables_dir = Path('pkl'), Path('tables/multi_task')

evaluate = Evaluate()

for model_path in sorted(models_dir.glob('*.h5')):
    model_name = model_path.stem

    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)

    # extract language and task from model name
    model_parts = model_name.split('_')
    lang, task = model_parts[0], model_parts[-1]

    # obtain the model's language and task
    lang = lang.upper() if lang in ('pt', 'es') else None
    task = task.upper() if task in ('pos', 'ner') else None

    # obtain the IDs of all datasets used in training
    dataset_ids = model.labelKeys.keys()
    dataset_ids_str = '_'.join(dataset_ids)

    # filename for fasttext word embeddings
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
        for task in dataset.tasks:
            # obtain mapping of indices to POS/NER labels
            label = task + '_BIO' if task == 'NER' else task
            idx2label = model.idx2Labels[label]

            # obtain correct and predicted sentences
            corr_idxs = [sentence[label] for sentence in test_data]
            pred_idxs = task_predictions[label]

            # convert indices to labels (POS tags or NER tags in BIO format)
            corr_labels = [[idx2label[idx] for idx in sent] for sent in corr_idxs]
            pred_labels = [[idx2label[idx] for idx in sent] for sent in pred_idxs]

            evaluate.eval(dataset.name, task, corr_labels, pred_labels, train_data, test_data)
            print(f'Evaluated dataset {dataset_id} - {task}')

evaluate.write_tables(tables_dir)
print(f'Wrote evaluation tables to {tables_dir}/')

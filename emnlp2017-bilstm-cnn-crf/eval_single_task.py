from pathlib import Path
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import loadDatasetPickle
from util.datasets import Dataset
from evaluator import Evaluator

# paths for input/output directories
models_dir, results_dir = Path('models/single_task'), Path('results/single_task')
pkl_dir, tables_dir = Path('pkl'), Path('tables/single_task')

evaluator = Evaluator()

for model_path in sorted(models_dir.glob('*.h5')):
    model_name = model_path.stem

    # obtain dataset ID and language
    dataset_id, task = model_name.rsplit('_', 1)
    task = task.upper()

    # obtain dataset name and task
    dataset = Dataset(dataset_id)
    dataset_name, lang = dataset.name, dataset.lang

    # filename for fasttext word embeddings
    embeddings_name = f'{lang.lower()}.fasttext.oov.vec'
    
    # load the dataset
    pkl_file = pkl_dir / f'{dataset_id}_{embeddings_name}.pkl'
    embeddings, mappings, data = loadDatasetPickle(pkl_file)

    # load the BiLSTM model
    model = BiLSTM.loadModel(model_path)
    model.setMappings(mappings, embeddings)

    # set the model dataset
    dataset_dict = dataset.to_dict(task)
    model.setDataset(dataset_dict, data)

    # obtain train and test data
    train_data = data[dataset_id]['trainMatrix']
    test_data = data[dataset_id]['testMatrix']

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
    print(f'Evaluated dataset {dataset_id} - {task}')

evaluator.write_tables(tables_dir)
print(f'Wrote evaluation tables to {tables_dir}/')

from pathlib import Path
from itertools import chain
import subprocess
import shutil

cwd = Path.cwd()
destination_dir = Path('../emnlp2017-bilstm-cnn-crf/data/')

# iterate over the Portuguese and Spanish datasets
for dataset_dir in chain(cwd.glob('pt_*'), cwd.glob('es_*')):
    dataset_name = dataset_dir.name
    data_dir = dataset_dir / 'data'

    # create dataset if it hasn't yet been created
    if not data_dir.exists():
        print('#### Creating dataset {} ####'.format(dataset_name))
        subprocess.run(['python', 'create_dataset.py'], cwd=str(dataset_dir))

    dest_dataset_dir = destination_dir / dataset_name
    if dest_dataset_dir.exists():
        shutil.rmtree(str(dest_dataset_dir))

    shutil.copytree(str(data_dir), str(dest_dataset_dir))

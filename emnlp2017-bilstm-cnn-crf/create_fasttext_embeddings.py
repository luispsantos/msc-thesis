from gensim.models.fasttext import load_facebook_vectors
from pathlib import Path
import pandas as pd
import gzip
import csv

lang = 'pt'
data_dir, embeddings_dir = Path('data'), Path('embeddings')

tokens = []
# read CoNLL formatted data from each dataset
for dataset_dir in sorted(data_dir.glob(f'{lang}_*')):
    for dataset_file in ['train', 'dev', 'test']:
        dataset_file = dataset_dir / (dataset_file + '.txt')

        # read only the column that contains the tokens
        token_col = pd.read_csv(dataset_file, sep='\t', names=['Token'],
                                usecols=[0], squeeze=True, quoting=csv.QUOTE_NONE)
        tokens.append(token_col)

# concatenate tokens from all datasets
tokens = pd.concat(tokens)

# sort tokens by frequency counts
token_counts = tokens.value_counts()
sorted_tokens = token_counts.index

# load FastText word vectors for the target language
model = load_facebook_vectors(embeddings_dir / f'cc.{lang}.300.bin')

# generate embeddings for all tokens (including OOV tokens)
embeddings = {token: model[token] for token in sorted_tokens if model[token].any()}

# create the embeddings file in Word2Vec format
with gzip.open(embeddings_dir / f'{lang}.fasttext.oov.vec.gz', 'wt') as embeddings_f:
    # write the header line in the first line
    embeddings_f.write(f'{len(embeddings)} 300\n')

    # write the token and the 300 dimensions embedding in each line
    for token, embedding in embeddings.items():
        embedding_str = ' '.join(f'{embedding_dim:1.4f}' for embedding_dim in embedding)
        embeddings_f.write(f'{token} {embedding_str}\n')


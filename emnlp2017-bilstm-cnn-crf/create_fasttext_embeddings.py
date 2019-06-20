from gensim.models.fasttext import load_facebook_vectors
from pathlib import Path
import pandas as pd
import gzip
import io
import csv
import yaml

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

# concatenate tokens from all datasets and lowercase tokens
tokens = pd.concat(tokens)
tokens = tokens.str.lower()

# sort tokens by frequency counts
token_counts = tokens.value_counts().sort_index(ascending=False) \
                                    .sort_values(ascending=False)
sorted_tokens = token_counts.index

# save token counts to a YAML file
with open(embeddings_dir / f'{lang}.fasttext.counts.yml', 'w') as counts_f:
    yaml.dump(token_counts.to_dict(), counts_f, allow_unicode=True, sort_keys=False)

# load FastText word vectors for the target language
model = load_facebook_vectors(embeddings_dir / f'cc.{lang}.300.bin')

# generate embeddings for all tokens (including OOV tokens)
embeddings = {token: model[token] for token in sorted_tokens if model[token].any()}

# create the compressed embeddings file in the word2vec format
with gzip.open(embeddings_dir / f'{lang}.fasttext.oov.vec.gz', 'wb') as gzip_f:
    with io.TextIOWrapper(io.BufferedWriter(gzip_f, 8*1024*1024), encoding='utf-8') as embeddings_f:
        # write the header line in the first line
        embeddings_f.write(f'{len(embeddings)} 300\n')

        # write the token and the 300 dimensions embedding in each line
        for token, embedding in embeddings.items():
            embedding_str = ' '.join(f'{embedding_dim:1.4f}' for embedding_dim in embedding)
            embeddings_f.write(f'{token} {embedding_str}\n')


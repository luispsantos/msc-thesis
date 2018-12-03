import urllib.request
from pathlib import Path
import lxml.html
import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvalidParamsError(Exception):
    """Raise when the word embeddings cannot be successfully loaded due to invalid parameters"""

class NILCPortugueseEmbeddings:

    def __init__(self, model='glove', model_type='cbow', model_dimension=50, embeddings_dir='embeddings'):
        """
        model: word2vec, wang2vec, glove, fasttext (case insensitive)
        model_type: sg (Skip-gram) or cbow (CBOW) (case insensitive)
        """
        available_models = ['word2vec', 'wang2vec', 'glove', 'fasttext']
        if model.lower() not in available_models:
            raise InvalidParamsError('Model must be one of: {}'.format(', '.join(available_models)))

        available_types = ['sg', 'cbow']
        if model_type.lower() not in available_types:
            raise InvalidParamsError('Model type must be one of: {}'.format(', '.join(available_types)))

        available_dimensions = [50, 100, 300, 600, 1000]
        if model_dimension not in available_dimensions:
            raise InvalidParamsError('Model dimension must be one of: {}'.format(', '.join(str(d) for d in available_dimensions)))

        self.lang = 'pt'
        self.model = model.lower()
        self.model_type = model_type.lower()
        self.model_dimension = model_dimension
        self.embeddings_dir = Path(embeddings_dir)
        
        if self.model == 'glove':
            embeddings_name = '{}_{}_{}d.gz'.format(self.lang, self.model, self.model_dimension)
        else:
            embeddings_name = '{}_{}_{}_{}d.gz'.format(self.lang, self.model, self.model_type, self.model_dimension)

        # download embeddings if these are not available locally
        self.path = self.embeddings_dir / embeddings_name
        if not self.path.exists():
            self.download()
        

    def download(self):
        # name of the embeddings file at the NILC website
        types = {'cbow': 'cbow', 'sg': 'skip'}
        if self.model == 'glove':
            embeddings_filename = '{}/{}_s{}.zip'.format(self.model, self.model, self.model_dimension)
        else:
            embeddings_filename = '{}/{}_s{}.zip'.format(self.model, types[self.model_type], self.model_dimension)

        # obtain URL to the embedding file by listing all URLs on the NILC website
        # since the download URL uses a specific machine and port, obtaining the links
        # directly from the NILC website is a way to future-proof the code
        nilc_doc = lxml.html.parse('http://nilc.icmc.usp.br/embeddings')
        nilc_urls = nilc_doc.xpath('//a/@href')
        embeddings_url = next(url for url in nilc_urls if url.endswith(embeddings_filename))

        logger.info('Downloading word embeddings')
        embeddings_zip_path = self.path.with_suffix('.zip')
        _, headers = urllib.request.urlretrieve(embeddings_url, embeddings_zip_path)

        logger.info('Extracting word embeddings')
        subprocess.run('''unzip -p {} |  # extracts embeddings text file from the ZIP archive
                          tail -n +2 |  # removes first line that contains stats (number of tokens + embedding dim)
                          gzip > {}  # compresses embeddings into a GZIP compressed text file
                       '''.format(embeddings_zip_path, self.path), shell=True)

        # remove the now unnecessary .zip file
        embeddings_zip_path.unlink()

        logger.info('Created embeddings file: {}'.format(self.path))

from lxml import html, etree
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import pandas as pd
import yaml

def process_table(table_df):
    # rename all the columns from Portuguese to English
    table_df.columns = ['Code', 'Filename', 'Author', 'Birthyear', 'Title', 'NumTokens',
    'TextEdited', 'TextStandardized', 'CompleteEdition', 'POSAnnot', 'SyntaxAnnot']

    # set the filename (document ID) as the index
    table_df.set_index('Filename', inplace=True)

    # remove unnecessary columns
    table_df.drop(columns=['Code', 'NumTokens', 'POSAnnot', 'SyntaxAnnot'], inplace=True)

    # remove parentheses from the birthyear column
    table_df.Birthyear = table_df.Birthyear.str.replace('[()]', '')

    # convert yes/no columns from Portuguese to English
    table_df.TextEdited.replace({'Editado': 'yes', 'Não-editado': 'no'}, inplace=True)
    table_df.TextStandardized.replace({'sim': 'yes', 'não': 'no'}, inplace=True)
    table_df.CompleteEdition.replace({'sim': 'yes', 'não': 'no'}, inplace=True)

    return table_df

# obtain corpus URL and input dataset directory from config file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

corpus_url, dataset_in_dir = config['url'], Path(config['dataset_in_dir'])
dataset_in_dir.mkdir(exist_ok=True)

# fetch the zip file of the POS annotated corpus
response = urllib.request.urlopen(corpus_url + 'texts/pos.zip')
corpus_bytes = response.read()

# extract all text files into a directory
corpus_zipfile = ZipFile(BytesIO(corpus_bytes))
corpus_zipfile.extractall(dataset_in_dir)

# fetch HTML content of the corpus inventory page
response = urllib.request.urlopen(corpus_url + 'catalogo.html')
html_content = response.read().decode('utf-8')

# parse HTML document and fetch table rows of text inventory 
root = html.document_fromstring(html_content)
rows = [tr for tr in root.xpath('//tr') if len(tr) >= 10]

# create a HTML text inventory table from the table rows
table_elem = etree.Element('table')
for row in rows: table_elem.append(row)

# create a DataFrame with the text inventory information
table_df = pd.read_html(etree.tostring(table_elem), header=0)[0]
table_df = process_table(table_df)

# keep metadata only for files which have POS annotations
corpus_filenames = [text_file.replace('_pos.txt', '') for text_file in corpus_zipfile.namelist()]
table_df = table_df.reindex(index=corpus_filenames)
table_df.to_csv('metadata.csv')

from lxml import html, etree
from pathlib import Path
from collections import OrderedDict
import urllib.request
import pandas as pd
import re
import yaml

def process_table(table_df):
    # merge columns Séc and Séc. and drop column Séc.
    table_df['Séc'].update(table_df['Séc.'])
    table_df['Séc'] = table_df['Séc'].astype(int)
    table_df.drop(columns=['Séc.'], inplace=True)

    # replace multiple spaces with a single space on column Texto
    table_df['Texto'] = table_df['Texto'].str.replace(' +', ' ')

    # drop column Notário since it's empty in more than 80% of texts
    table_df.drop(columns=['Notário'], inplace=True)

    # convert numerical columns to string but retain NaN values
    century, date = table_df['Séc'], table_df['Data']
    century.where(pd.isna(century), century.astype(str), inplace=True)
    date.where(pd.isna(date), date.astype(str), inplace=True)

    return table_df

def process_text(text):
    text = text.replace('\r\n', ' ')    # replace Windows-style newlines with a space
    text = text.replace('\xa0', ' ')    # replace non-breaking space of ISO-8859-1 with a space
    text = re.sub(' +', ' ', text)      # replace multiple consecutive spaces with a space

    return text

# obtain corpus URL and input dataset directory from config file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

corpus_url, dataset_in_dir = config['url'], Path(config['dataset_in_dir'])
dataset_in_dir.mkdir(exist_ok=True)

table_elems, text_elems = [], []
corpus_section = '//div[@class="Section1"]'

# iterate through multiple sections of the annotated corpus
for idx in range(101, 107):
    # fetch HTML content of corpus page
    response = urllib.request.urlopen(f'{corpus_url}/gencontent.jsp?id={idx}')
    html_content = response.read().decode('utf-8')

    # parse HTML document with lxml
    parser = html.HTMLParser(encoding='ISO-8859-1')
    root = html.document_fromstring(html_content, parser=parser)

    # extract HTML elements of the tables and corpus text
    table_elems += root.xpath(corpus_section + '/table')
    text_elems += root.xpath(corpus_section + '/p')

# create a single DataFrame from the HTML content of the tables
# these tables contain header information about the corpus text
tables_html = [etree.tostring(table_elem, encoding=str) for table_elem in table_elems]
tables_df = pd.read_html(''.join(tables_html), header=0, index_col='Documento')

table_df = pd.concat(tables_df, axis=0, sort=False)
table_df = process_table(table_df)
table_df.to_csv('metadata.csv')

# obtain list of corpus texts and filter out empty texts
corpus_text = [text_elem.text_content() for text_elem in text_elems]
corpus_text = [text.strip() for text in corpus_text if text.strip()]
corpus_text = [process_text(text) for text in corpus_text]

# create a text file per document with a header and text content
for (doc_id, header), text in zip(table_df.iterrows(), corpus_text):
    text_file = dataset_in_dir / (doc_id + '.txt')
    
    # choose which attributes to display on header line and discard 
    # empty attributes. An OrderedDict preserves attribute order.
    attrs = header.dropna().to_dict(OrderedDict)
    
    # create a <text> XML header line for each text file
    header_elem = etree.Element('text', attrs)
    header_line = etree.tostring(header_elem, encoding=str)
    
    # write header line and corpus text to the text file
    text_file.write_text(header_line + '\n' + text + '\n')


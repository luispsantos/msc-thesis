from lxml import html, etree
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from os.path import getsize
from multiprocessing import Pool
import urllib.request
import pandas as pd
import yaml

def get_text(elem, query, nsmap):
    return elem.xpath(query, namespaces=nsmap)[0].text

def parse_xml_file(xml_file, parser=etree.XMLParser(recover=True), nsmap={'tei': 'http://www.tei-c.org/ns/1.0'}):
    # parse XML document
    doc = etree.parse(str(xml_file), parser)
    root = doc.getroot()

    # obtain the name of the archive where the document is stored
    archive = get_text(root, '//tei:msDesc//tei:institution', nsmap)

    # obtain element with sender and receiver metadata
    sender, receiver = root.xpath('//tei:correspAction', namespaces=nsmap)

    # fetch author and addressee names
    author = get_text(sender, 'tei:persName/tei:name', nsmap)
    addressee = get_text(receiver, 'tei:persName/tei:name', nsmap)

    # fetch geographic names of origin and destination
    origin = get_text(sender, 'tei:placeName', nsmap)
    destination = get_text(receiver, 'tei:placeName', nsmap)

    # fetch the date and the year (discard month and day information)
    date = sender.xpath('tei:date/@when-custom', namespaces=nsmap)[0]
    year = date[:date.index('.')] if '.' in date else date  # e.g. 1689.11.08, 1580-1600.

    # fetch the topic and type of the letter
    category_xpath = '//tei:category[@xml:id="{}"]/tei:catDesc'.format
    letter_topic = get_text(root, category_xpath('pragmatics'), nsmap)
    letter_type = get_text(root, category_xpath('type'), nsmap)

    # fetch list of keywords for the letter
    keywords = root.xpath(category_xpath('socioHistoricalSource') + '/text()[last()]', namespaces=nsmap)
    keywords = keywords[0] if keywords else None

    doc_metadata = {'ID': xml_file.stem, 'Archive': archive, 'Author': author, 'Addressee': addressee,
                    'Origin': origin, 'Destination': destination, 'Year': year, 'Topic': letter_topic,
                    'Type': letter_type, 'Keywords': keywords}

    return doc_metadata

def process_table(table_df):
    # set the document ID as the index
    table_df.set_index('ID', inplace=True)

    # apply a specific ordering for the columns
    table_df = table_df[['Archive', 'Year', 'Type', 'Topic', 'Origin',
                        'Destination', 'Author', 'Addressee', 'Keywords']]

    # remove extra whitespace on some columns
    table_df.Origin = table_df.Origin.str.rstrip()
    table_df.Destination = table_df.Destination.str.rstrip()
    table_df.Keywords = table_df.Keywords.str.lstrip()

    return table_df

# obtain corpus URL and input dataset directory from config file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

corpus_url, dataset_in_dir = config['url'], Path(config['dataset_in_dir'])
dataset_in_dir.mkdir(exist_ok=True)

# download POS annotated corpus and XML corpus to extract metadata
pos_dir, xml_dir = dataset_in_dir / 'POS', dataset_in_dir / 'XML'
centuries = [1500, 1600, 1700, 1800]

for century in centuries:
    pos_filename, xml_filename = f'PT{century}_POS', f'PT{century}_XML-TEI_P5'

    # fetch the zip file of the POS annotated corpus
    response = urllib.request.urlopen(f'{corpus_url}/files/{pos_filename}.zip')
    pos_corpus_bytes = response.read()

    # extract all text files into POS directory
    pos_corpus_zipfile = ZipFile(BytesIO(pos_corpus_bytes))
    pos_corpus_zipfile.extractall(pos_dir / pos_filename)

    # fetch the zip file of the XML documents
    response = urllib.request.urlopen(f'{corpus_url}/files/{xml_filename}.zip')
    xml_corpus_bytes = response.read()

    # extract all XML files into XML directory
    xml_corpus_zipfile = ZipFile(BytesIO(xml_corpus_bytes))
    xml_corpus_zipfile.extractall(xml_dir / xml_filename)

# parse letter metadata from non-empty XML files
with Pool() as pool:
    xml_files = filter(getsize, sorted(xml_dir.rglob('*.xml')))
    metadata = pool.map(parse_xml_file, xml_files)

# output extracted XML metadata in tabular form
table_df = pd.DataFrame(metadata)
table_df = process_table(table_df)
table_df.to_csv('metadata.csv')

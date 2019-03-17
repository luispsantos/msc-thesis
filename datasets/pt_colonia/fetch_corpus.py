from lxml import html, etree
from pathlib import Path
import urllib.request
import urllib.error
import pandas as pd
import yaml

def process_table(table_df):
    # set the filename (document ID) as the index
    table_df.Filename = table_df.Filename.str.replace(',txt$', '')
    table_df.set_index('Filename', inplace=True)

    # insert a new column for the century in a 2 digit format
    table_df.insert(2, 'Century', table_df.Subcorpus.str[:2].astype(int))

    # remove unnecessary columns Subcorpus and Tokens
    table_df.drop(columns=['Subcorpus', 'Tokens'], inplace=True)

    # replace dash symbol in year ranges
    table_df.Year = table_df.Year.str.replace(' – ', '-')

    # use dash symbol to represent missing data
    table_df.Year = table_df.Year.str.replace('.+Century$', '-')

    return table_df

def process_text(text):
    # there should be exactly one newline at the end of the file
    text = text[:-1] if text.endswith('\n\n') else text
    text = text + '\n' if not text.endswith('\n') else text

    return text

def decode(content, encodings=('utf-8-sig', 'latin1')):
    # utf-8-sig correctly decodes files with a BOM character
    # fallback to latin1 encoding if decoding as utf-8 fails
    for encoding in encodings:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise RuntimeError('Unable to find right encoding')

# obtain corpus URL and input dataset directory from config file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

corpus_url, dataset_in_dir = config['url'], Path(config['dataset_in_dir'])
dataset_in_dir.mkdir(exist_ok=True)

# fetch HTML content of the corpus inventory page
response = urllib.request.urlopen(corpus_url + 'inventory.html')
html_content = response.read().decode('utf-8')

# parse HTML document and fetch text inventory table
root = html.document_fromstring(html_content)
table_elem = root.xpath('//div[@id="content"]/table')[0]

# create a DataFrame with the text inventory information
table_df = pd.read_html(etree.tostring(table_elem))[0]
table_df = process_table(table_df)

# create century directories in order to organize text files by century
table_df['CenturyDir'] = (table_df.Century-1).astype(str) + '00_POS'
for century_dir in table_df.CenturyDir.unique():
    Path(dataset_in_dir, century_dir).mkdir(exist_ok=True)

# obtain the text file URLs
file_urls = table_elem.xpath('tr/td/a/@href')
unavailable_files = []

# download each text file and store these on the input dataset directory
for (filename, header), file_url in zip(table_df.iterrows(), file_urls):
    text_file = Path(dataset_in_dir, header.CenturyDir, filename + '.txt')
    try:
        response = urllib.request.urlopen(corpus_url + file_url)
        text = decode(response.read())
        if text.startswith('<text'):
            text = process_text(text)
            text_file.write_text(text)
        else:
            print(f'{filename} - Invalid textual format. Skipping ...')
            unavailable_files.append(filename)
    except urllib.error.HTTPError as e:
        print(f'{filename} - {e}. Skipping ...')
        unavailable_files.append(filename)

# keep metadata only for available files
table_df.drop(index=unavailable_files, columns='CenturyDir', inplace=True)
table_df.to_csv('metadata.csv')

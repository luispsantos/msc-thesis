from pathlib import Path
import pandas as pd
import re
import unicodedata
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing util package from parent directory
sys.path.insert(1, str(Path.cwd().parent))
from util import *

# read variables from the configuration file
config = load_yaml('config.yml')
dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
rules = load_yaml('rules.yml')
pos_tagset_ud_map, token_upos_map = rules['pos_tagset_ud_map'], rules['token_upos_map']

token_re = re.compile('^(?P<Token>.+?)_(?P<POS>[A-Z+-]+)$')

def handle_contraction(match):
    # obtain contraction groups (e.g. [[da]]DE=_P=A_AD -> token=da, tags=DE=_P=A_AD)
    token, tags = match.group('token'), match.group('tags')

    # split tags into lemmas and POS tags (e.g. DE=_P=A_AD -> [['DE', 'P'], ['A, 'AD']])
    tags = tags.replace('=_', '_')
    tag_list = [tag.split('_') for tag in tags.split('=')]

    # concatenate POS tags and return a new contraction token (e.g. da_P+AD)
    lemmas, pos = zip(*tag_list)
    contraction = '_'.join([token, '+'.join(pos)])

    return contraction

def preprocess_text(text):
    # detach the header line from the text content
    header, text = text.splitlines()

    # remove section meta tags (e.g. <F 159rB>, <F 118vA>, <pf 815>)
    text = re.sub('<.+?>', '', text)

    # remove excerpts in Latin (e.g. {In d(e)j n(omi)ne am(en).})
    text = re.sub('{.+?}', '', text)

    # the dollar symbol - $ and (($)) - indicate sentence boundaries
    text = re.sub('\$|\({2}\$\){2}', '.', text)

    # replace consecutive dots (sentence boundaries) by a single dot
    text = re.sub('\. \.', '.', text)

    # remove comments with meta information (e.g. ((Livro I, fl. 9r AB)))
    text = re.sub('\({2}.+?\){2}', '', text)

    # normalize base form of contractions (e.g. [[da]]DE=_P=A_AD -> da_P+AD)
    text = re.sub('\[{2}(?P<token>.+?)\]{1,2}(?P<tags>[A-Z]\S+)', handle_contraction, text)

    # remove duplicate words when followed by a [sic] (e.g. de_P de_P [sic] -> de_P)
    text = re.sub(r'(?i)\b([\w()]+) \1 \[sic\]', r'\1', text)

    # remove [sic] token with the meaning that the previous word contains a non-corrected error
    # these errors are minor and occur too infrequently to justify writing exception rules
    text = re.sub('\[sic\]', '', text)

    # remove [...] and (...) corresponding to excerpts deliberately omitted by the editors
    text = re.sub('[\[(]\.{3}[\])]', '', text)

    # remove square brackets and parentheses corresponding to grapheme corrections and abbreviations
    text = re.sub('[[\]()]', '', text)
    
    # remove forward slash when it corresponds to full grapheme corrections (e.g. /no/sso -> nosso)
    text = re.sub('\/([^.? ]+?)\/', r'\1', text)

    # remove forward slash and inner content on partial grapheme corrections (e.g. /.../, /Sa..ne/)
    text = re.sub('\/.+?\/', '', text)

    # separate punctuation from POS tags (e.g. sempre_NC. -> sempre_NC ., preuedo_A; -> preuedo_A ;)
    text = re.sub('(_\S+?)([.|,|:|;])', r'\1 \2', text)

    # remove question marks from POS tags, which express tag uncertainty (e.g. Caber_VINF? -> Caber_VINF)
    text = re.sub('(_\S+?)([?])', r'\1', text)

    # convert circumflex accent ´c´ form to c^ (e.g. a´a´s -> a^s, home´e´s -> home^s, va´a´sco -> va^sco)
    text = re.sub(r'(\w)´~?\1~?´|(i)´j´', r'\1^', text)

    # case where letters are different (e.g. lixbo´a´ -> lixbo^a, Jo´a´nes -> Jo^anes, mo´e´da -> mo^eda)
    text = re.sub(r'´(\w)´', r'^\1', text)

    # convert diacritical marks to combining diacritical marks (e.g. a` -> à, o´ -> ó, e^ -> ê, a~ -> ã)
    diacritics = {'`': '\u0300', '´': '\u0301', '^': '\u0302', '~': '\u0303'}
    text = text.translate(str.maketrans(diacritics))
    text = unicodedata.normalize('NFC', text)

    # remove the only token without POS tag (since all tokens should have an associated POS tag)
    text = re.sub('díz ', '', text)

    return text

def text_to_dataframe(text):
    # split the text on punctuation symbols
    raw_sents = re.split('( (?:\.|,|:|;\.?|-))', text)

    sents = []
    # concatenate the sentences with the punctuation symbols/sentence boundaries
    for sent in raw_sents:
        if '_' in sent:
            sents.append(sent)
        elif sent.lstrip():
            sents[-1] += sent + '_PUNC'

    data_df = read_data(sents, token_re)
    return data_df

data_df = read_text_files(dataset_in_dir, preprocess_text, text_to_dataframe)

data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)
replace_values(token_upos_map, data_df.Token, data_df.UPOS)

# split data into train, dev and test sets and write data to disk
train_test_dfs = train_test_split(data_df)
write_data(train_test_dfs, dataset_out_dir, output_columns)


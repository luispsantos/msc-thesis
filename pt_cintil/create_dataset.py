from lxml import etree
from html import escape
from pathlib import Path
import random
import re

# fix the random seed for reproducibility
random.seed(123)

dataset_in_dir = Path('original_data')
dataset_out_dir = Path('data')

train_proportion = 0.8
dev_proportion = 0.1

cintil_input_file = dataset_in_dir / 'CINTIL-WRITTEN.txt'

# create output dataset directory if it doesn't exist
if not dataset_out_dir.exists():
    dataset_out_dir.mkdir(parents=True)

output_columns = ['Token', 'POS', 'NER']
output_separator = ' '
token_re = re.compile(r'(?:\\\*)?(?P<Token>.+?)(?:\*/)?/(?:.+?/)?(?P<POS>.+?)\d?(?:#.+?)?\[(?P<NER>.+?)\]')

pos_tagset_ud_map = {
    'ADJ': 'ADJ',
    'ADV': 'ADV',
    'CARD': 'NUM',
    'CJ': 'CCONJ',
    'CL': 'PRON',
    'CN': 'NOUN',
    'DA': 'DET',
    'DEM': 'PRON',
    'DFR': 'NUM',
    'DGT': 'NUM',
    'DGTR': 'NUM',
    'DM': 'INTJ',
    'EOE': 'ADV',
    'GER': 'VERB',
    'IA': 'DET',
    'IND': 'PRON',
    'INF': 'VERB',
    'INT': 'PRON',
    'ITJ': 'INTJ',
    'LADV': 'ADV',
    'LCJ': 'CCONJ',
    'LCN': 'NOUN',
    'LDEM': 'PRON',
    'LDFR': 'NUM',
    'LITJ': 'INTJ',
    'LPREP': 'ADP',
    'LPRS': 'PRON',
    'LREL': 'PRON',
    'LTR': 'SYM',
    'MGT': 'NOUN',
    'MTH': 'NOUN',
    'ORD': 'ADJ',
    'PADR': 'NOUN',
    'PNM': 'PROPN',
    'PNT': 'PUNCT',
    'POSS': 'DET',
    'PP': 'ADV',
    'PPA': 'ADJ',
    'PREP': 'ADP',
    'PPT': 'VERB',
    'PRS': 'PRON',
    'QNT': 'DET',
    'REL': 'PRON',
    'STT': 'NOUN',
    'SYB': 'SYM',
    'UM': 'DET',
    'V': 'VERB',
    'VAUX': 'AUX',
    'WD': 'NOUN'
}

# read CINTIL's file contents
cintil_text = cintil_input_file.read_text()

# delete unnecessary tags: <i>, </i>, <t>, </t>
# their function is not fundamental to the creation of POS or NER systems
# furthermore some of these tags are opened without being properly closed
cintil_text = re.sub(r'</?(i|t)> ', '', cintil_text)

# URL encode '&' symbol - necessary in order to use a XML parser
cintil_text = cintil_text.replace('&', '&amp;')

# URL encode '<' and '>' symbols whenever they occur as tokens (not as XML tags)
cintil_text = re.sub(r'(?:\\\*)?(<|>)(?:\*/)?/[^>]+?\[.+?\]', lambda match: escape(match.group(0)), cintil_text)

# parse XML
cintil = etree.fromstring(cintil_text)

# find XML tags which correspond to sentences
raw_sents = cintil.xpath('excerpt/text/p/s/text()')
sents = []

for idx, raw_sent in enumerate(raw_sents):
    raw_tokens = raw_sent.strip().split(' ')
    sent_tokens = []

    for raw_token in raw_tokens:
        match = token_re.match(raw_token)
        token_dict = match.groupdict()
        token_dict['Token'] = token_dict['Token'][:-1] if token_dict['Token'][-1] == '_' else token_dict['Token']

        #make sure the match included the whole token
        assert match.start() == 0 and match.end() == len(raw_token), \
               'Token regex did not cover the whole token: {}'.format(raw_token)

        sent_tokens.append(token_dict)

    sents.append(sent_tokens)

# randomize the dataset before splitting
random.shuffle(sents)

# create train-dev-test split
num_sents = len(sents)
dataset_sents = {}

train_split = int(train_proportion * num_sents)
dev_split = int((train_proportion + dev_proportion) * num_sents)

dataset_sents['train'] = sents[:train_split]
dataset_sents['dev'] = sents[train_split:dev_split]
dataset_sents['test'] = sents[dev_split:]

for dataset_type in ['train', 'dev', 'test']:
    cintil_output_file = dataset_out_dir / '{}.txt'.format(dataset_type)
    sents = dataset_sents[dataset_type]

    with cintil_output_file.open('w') as f:
        for sent_tokens in sents:
            for token_idx, token_dict in enumerate(sent_tokens):
                token_pos_cintil = token_dict['POS']

                #map POS tags from CINTIL tagset to UD tagset
                if token_dict['Token'] == 'sido' and token_pos_cintil == 'PPT':
                    token_pos_ud = 'AUX'
                # discard tokens with optional gender and number (e.g., (s), (as), etc.)
                elif token_pos_cintil == 'TERMN':
                    continue
                else:
                    token_pos_ud = pos_tagset_ud_map[token_pos_cintil] if token_dict['POS'] in pos_tagset_ud_map else token_dict['POS']

                token_dict['POS'] = token_pos_ud
                token_columns = [token_dict[col] for col in output_columns]

                token_line = output_separator.join(token_columns)
                f.write(token_line + '\n')

            # write an empty line to denote sentence boundaries
            f.write('\n')

    print('Created file {}'.format(cintil_output_file))

from pathlib import Path
import pandas as pd
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing util package from parent directory
sys.path.insert(1, str(Path.cwd().parent))
from util import *

#read variables from the configuration file
config = load_yaml('config.yml')
dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
rules = load_yaml('rules.yml')
pos_tagset_ud_map, ner_tagset_map = rules['pos_tagset_ud_map'], rules['ner_tagset_map']

# loading dict of MWEs and their UPOS tags extracted from AnCora UD
mwe_upos_map = load_yaml('mwes.yml')

def read_conll09(data_in_path):
    data_df = pd.read_csv(data_in_path, sep='\t', names=['Token', 'POS_0', 'Feat', 'Ents'],
                          usecols=[1, 4, 6, 12], skip_blank_lines=False,
                          quoting=csv.QUOTE_NONE, comment='#')
    return data_df

def compute_prefix(ent):
    """
    Compute whether token corresponds to a single, start, or
    end entity span. E.g. (loc) -> S, (org -> B, org) -> E.
    """
    if ent[0] == '(' and ent[-1] == ')':
        prefix = 'S'
    elif ent[0] == '(':
        prefix = 'B'
    elif ent[-1] == ')':
        prefix = 'E'
    else:
        prefix = ''

    return prefix

def expand_mwes(data_df, mwe_upos_map):
    """
    Expands all the MWEs, whereas MWEs on the AnCora corpus
    are represented as a single token (e.g. mayo_del_2002),
    these MWEs are expanded to span multiple tokens. The POS
    tags are not available at the token-level but only at the
    MWE-level. So we map the UPOS tags from the MWEs used by
    the AnCora UD corpus as AnCora UD contains the same MWEs.
    """
    # categorize tokens as either MWEs or non-MWEs
    mwe_mask = data_df.Token.str.contains('_', na=False)
    mwes, non_mwes = data_df[mwe_mask], data_df[~mwe_mask]

    # obtain the individual tokens and UPOS tags for each MWE
    mwe_tokens = mwes.Token.str.split('_')
    mwe_upos = mwes.Token.map(mwe_upos_map).str.split('_')

    # find number of tokens per MWE and expand MWEs
    mwe_size = mwe_tokens.str.len()
    expanded_mwes = mwes.loc[mwes.index.repeat(mwe_size)]

    # assign expanded MWE tokens and UPOS tags
    expanded_mwes.Token = flatten_list(mwe_tokens)
    expanded_mwes.UPOS = flatten_list(mwe_upos)

    # assign entity tags to expanded MWEs
    expanded_mwes.Ents = compute_mwe_ents(mwes, mwe_size)

    # join together MWE and non-MWE tokens
    data_df = pd.concat([expanded_mwes, non_mwes], axis=0)

    # merge MWE and non-MWE tokens by sorting the index
    data_df.sort_index(kind='mergesort', inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    return data_df

def compute_mwe_ents(mwes, mwe_size):
    """
    Compute the entities for expanded MWEs. For example,
    single entities are converted to start and end ents
    (e.g. 25_años+(num) -> 25+(num años+num) ). Start
    and end entities only occur at the initial and final
    token respectively (e.g. Mitsubishi_Corporation+org)
    -> Mitsubishi+O Corporation+org) ).
    """
    mwe_start_ents, mwe_end_ents = [], []
    for token_ents in mwes.Ents:
        # find all start and end entities for each MWE
        start_ents, end_ents = [], []
        for ent in token_ents.split('|'):
            prefix = compute_prefix(ent)
            if prefix == 'B':
                start_ents.append(ent)
            elif prefix == 'E':
                end_ents.append(ent)
            elif prefix == 'S':
                start_ents.append(ent.rstrip(')'))
                end_ents.append(ent.lstrip('('))

        # obtain a string with the start and end entities
        mwe_start_ents.append('|'.join(start_ents))
        mwe_end_ents.append('|'.join(end_ents))

    mwe_ents = []
    for start_ents, end_ents, size in zip(mwe_start_ents, mwe_end_ents, mwe_size):
        # apply start and end entities at the initial and final tokens
        # of the expanded MWE, and empty entity (_) everywhere else
        start_ents = start_ents if start_ents else '_'
        end_ents = end_ents if end_ents else '_'
        mwe_ents += [start_ents] + ['_'] * (size-2) + [end_ents]

    return mwe_ents

def find_ents(data_df, ner_tagset_map):
    """
    Finds all named entities on AnCora, in the form of
    entity types and start and end ranges. Entities can
    be of the kind B, E, S (i.e. begin, end, single), and
    tokens may have multiple of said entities. Additionally,
    entities can be nested, so we use a LIFO data structure
    (i.e. a stack) to keep track of the nesting depth.
    """
    # obtain the tokens which contain entities
    ent_mask = data_df.Ents.str.contains('[()]', na=False)
    ents = data_df.Ents[ent_mask]

    # iterate through tokens with entities, where
    # tokens may have multiple entities which can
    # be of the kind B, E, S (begin, end, single).
    ent_list, ent_stack = [], []
    for idx, token_ents in ents.iteritems():
        for ent in token_ents.split('|'):
            # obtain prefix and ent_type (e.g. (org -> B-org)
            prefix = compute_prefix(ent)
            ent_type = ent.strip('()')
            ent_type = ner_tagset_map[ent_type.upper()]

            if prefix == 'B':
                # push entity start into the stack
                ent_stack.append((idx, ent_type))
            elif prefix == 'E':
                # pop last entity from stack
                start_idx, start_ent = ent_stack.pop()
                if start_ent != ent_type:
                    # sometimes the entity match occurs with the
                    # penultimate element rather than the last, in
                    # those cases push the last entity into the stack
                    last_ent = (start_idx, start_ent)
                    start_idx, start_ent = ent_stack.pop()
                    ent_stack.append(last_ent)

                ent_list.append((start_idx, idx, len(ent_stack), ent_type))
            elif prefix == 'S':
                ent_list.append((idx, idx, len(ent_stack), ent_type))

    return ent_list

def ents_to_bio(ent_list):
    """
    Converts a list of entity types and token spans
    to BIO-formatted entities.
    """
    # obtain only the outer entities
    outer_ents = [(start_idx, end_idx, ent_type)
                  for start_idx, end_idx, stack_depth, ent_type
                  in ent_list if stack_depth == 0]

    ent_idx, ent_bio = [], []
    for start_idx, end_idx, ent_type in outer_ents:
        ent_idx += range(start_idx, end_idx+1)
        ent_bio += [f'B-{ent_type}'] + [f'I-{ent_type}'] * (end_idx-start_idx)

    return ent_idx, ent_bio

conll_files = {'train': 'es.train', 'dev': 'es.devel', 'test': 'es.test'}

for dataset_type, conll_file in conll_files.items():
    data_in_path = dataset_in_dir / conll_file
    data_out_path = dataset_out_dir / (dataset_type + '.txt')

    # read CoNLL-2009 data
    data_df = read_conll09(data_in_path)

    # remove tokens without available text
    data_df = data_df[data_df.Token != '_']

    # obtain the first letter of the initial feature
    # (e.g. postype=article -> a, postype=proper -> p)
    first_feat = data_df.Feat.str.partition('=', False).str[2].str[0]

    # concatenate POS (general tag) and first feature (specific tag)
    data_df['POS_0+1'] = data_df['POS_0'] + first_feat

    # use uppercase versions of the POS tags
    data_df['POS_0'] = data_df['POS_0'].str.upper()
    data_df['POS_0+1'] = data_df['POS_0+1'].str.upper()

    # map POS tagset to UD tagset based on the first and first+second characters of POS tags
    data_df['UPOS'] = data_df['POS_0'].map(pos_tagset_ud_map)
    replace_values(pos_tagset_ud_map, data_df['POS_0+1'], data_df.UPOS)

    # expand MWEs into separate tokens
    data_df = expand_mwes(data_df, mwe_upos_map)

    # find all entities and convert them to BIO format
    ent_list = find_ents(data_df, ner_tagset_map)
    ent_idx, ent_bio = ents_to_bio(ent_list)

    # assign the BIO-formatted entities to the NER column
    data_df.loc[data_df.Ents.notna(), 'NER'] = 'O'
    data_df.loc[ent_idx, 'NER'] = ent_bio

    # write data to disk
    write_data(data_df, data_out_path, output_columns)


from pathlib import Path
from itertools import groupby
from operator import itemgetter
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

            if ent_type == 'O':
                # if the entity type is mapped to O then
                # there is no need to push it into the stack
                continue
            elif prefix == 'B':
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

def create_ner_column(data_df, ent_list):
    """
    Creates a NER column containing the entity types
    from a list of nested entity types and token spans.
    """
	# sort entity list by stack depth
    ent_list = sorted(ent_list, key=itemgetter(2))

    for stack_depth, ents in groupby(ent_list, itemgetter(2)):
        ent_idxs, ent_types = [], []
		# populate lists with the token indexes that
		# make part of entities for each stack level
        for start_idx, end_idx, _, ent_type in ents:
            ent_idxs += range(start_idx, end_idx+1)
            ent_types += [ent_type] * (end_idx+1 - start_idx)
            
        # assign entity types to NER column
        data_df.loc[ent_idxs, 'NER'] = ent_types

    # tokens which are not named entities or proper nouns are tagged as O
    outside_mask = data_df.NER.isna() | (data_df.UPOS != 'PROPN')
    data_df.loc[data_df.Token.notna() & outside_mask, 'NER'] = 'O'

def expand_mwes(data_df, mwe_upos_map):
    """
    Expands all the MWEs, whereas MWEs on the AnCora corpus
    are represented as a single token (e.g. mayo_del_2002),
    these MWEs are expanded to span multiple tokens. The POS
    tags are not available at the token-level but only at the
    MWE-level. So we map the UPOS tags from the MWEs used by
    the AnCora UD corpus as AnCora UD contains the same MWEs.
    Entity types are mapped to the BIO scheme in the process.
    """
    # categorize tokens as either MWEs or non-MWEs
    mwe_mask = data_df.Token.str.contains('_', na=False)
    mwes, non_mwes = data_df[mwe_mask], data_df[~mwe_mask].copy()

    # obtain the individual tokens and UPOS tags for each MWE
    mwe_tokens = mwes.Token.str.split('_')
    mwe_upos = mwes.Token.map(mwe_upos_map).str.split('_')

    # find number of tokens per MWE and expand MWEs
    mwe_size = mwe_tokens.str.len()
    expanded_mwes = mwes.loc[mwes.index.repeat(mwe_size)]

    # assign expanded MWE tokens and UPOS tags
    expanded_mwes.Token = flatten_list(mwe_tokens)
    expanded_mwes.UPOS = flatten_list(mwe_upos)

    # compute the entities for expanded MWEs featuring the BIO scheme
    mwe_ents = [['O'] * ent_size if ent_type == 'O' else
                [f'B-{ent_type}'] + [f'I-{ent_type}'] * (ent_size-1)
                for ent_type, ent_size in zip(mwes.NER, mwe_size)]

    # assign entity tags in BIO scheme to expanded MWEs
    expanded_mwes.NER = flatten_list(mwe_ents)

    # insert the "B-" prefix for non-MWE tokens belonging to a named entity
    non_mwes.NER.mask(non_mwes.NER != 'O', 'B-' + non_mwes.NER, inplace=True)

    # join together MWE and non-MWE tokens
    data_df = pd.concat([expanded_mwes, non_mwes], axis=0)

    # merge MWE and non-MWE tokens by sorting the index
    data_df.sort_index(kind='mergesort', inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    return data_df

def process_dataset(data_in_path):
    # read CoNLL-2009 data
    data_df = read_conll09(data_in_path)

    # remove tokens without available text
    data_df = data_df[data_df.Token != '_']
    data_df.reset_index(drop=True, inplace=True)

    # fix erroneous underscore at the end of MWEs
    data_df.Token = data_df.Token.str.rstrip('_')

    # obtain the first letter of the initial feature
    # (e.g. postype=article -> a, postype=proper -> p)
    first_feat = data_df.Feat.str.partition('=', False).str[2].str[0]

    # concatenate POS (general tag) and first feature (specific tag)
    data_df['POS_0+1'] = data_df['POS_0'] + first_feat

    # use uppercase versions of the POS tags
    data_df['POS_0'] = data_df['POS_0'].str.upper()
    data_df['POS_0+1'] = data_df['POS_0+1'].str.upper()

    # convert contracted prepositions to SC (ADP+DET) tag
    contracted_mask = data_df.Feat.str.endswith('contracted=yes', na=False)
    data_df.loc[(data_df['POS_0+1'] == 'SP') & contracted_mask, 'POS_0+1'] = 'SC'

    # map POS tagset to UD tagset based on the first and first+second characters of POS tags
    data_df['UPOS'] = data_df['POS_0'].map(pos_tagset_ud_map)
    replace_values(pos_tagset_ud_map, data_df['POS_0+1'], data_df.UPOS)

    # find all entities and create a column containing the entity types
    ent_list = find_ents(data_df, ner_tagset_map)
    create_ner_column(data_df, ent_list)

    # expand MWEs into separate tokens featuring BIO-formatted entities
    data_df = expand_mwes(data_df, mwe_upos_map)

    return data_df

# process dataset with pre-made data splits and write data to disk
data_in_files = {'train': 'es.train', 'dev': 'es.devel', 'test': 'es.test'}
dataset_with_splits(process_dataset, data_in_files, dataset_in_dir, dataset_out_dir, output_columns)

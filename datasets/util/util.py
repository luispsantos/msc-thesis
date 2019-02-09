import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import csv
import random

# fix the random seed for reproducibility
random.seed(123)

SENT_BOUNDARY = np.nan
IS_SENT_BOUNDARY = lambda val: pd.isna(val)
NOT_SENT_BOUNDARY = lambda val: pd.notna(val)

def read_data(raw_tokens, token_re):
    """
    Creates a DataFrame containing the tokens as the rows and token information
    such as the PoS/NER tag or lemma of the token as columns. To indicate sentence
    boundaries the special token SENT_BOUNDARY should be used between sentences.

    :param raw_tokens: An iterable (typically a generator function called over the
    input dataset file) that yields tokens and yields SENT_BOUNDARY between sentences.
    :param token_re: A regex to extract the token attributes from the raw token by
    means of capture groups. Each capture group in the regex becomes a new column.
    """
    # create a Series of raw tokens
    raw_tokens = pd.Series(raw_tokens)

    # make sure the token regex matches the whole token
    assert raw_tokens.str.match(token_re).all(), 'Token regex failed to match at least one token'

    # convert a Series of raw tokens into a DataFrame of cleaned-up strings
    # where each capture group names in the regex are used as column names
    data_df = raw_tokens.str.extract(token_re)

    return data_df

def read_conllu(dataset_in_path, keep_columns=['ID', 'FORM', 'UPOS']):
    """
    Reads a CoNLL-U file into a DataFrame. Optionally discards some of the columns.
    The CoNLL-U format specification can be found at http://universaldependencies.org/format.html.
    """
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    data_df = pd.read_csv(dataset_in_path, sep='\t', names=column_names,
                          skip_blank_lines=False, quoting=csv.QUOTE_NONE, comment='#')

    # keep only a few columns of the dataset and discard the rest
    data_df = data_df[keep_columns]
    data_df.rename(columns={'FORM': 'Token'}, inplace=True)

    return data_df

def extract_multiwords(data_df, multiword_upos_map=None):
    """
    Extracts multi-word tokens from a CoNLL-U file. Multi-word tokens are defined in CoNLL-U by range IDs.
    Removes all tokens that are included in range IDs. The UPOS tags of those tokens are concatenated
    to derive a UPOS tag for the multi-word token (e.g. de_ADP os_DET -> dos_ADP+DET). If defined,
    multiword_upos_map provides a mapping between concatenated tags and single tags (e.g. ADP+ADV -> ADV).
    """
    # compute tokens with range IDs (e.g. 7-8) which define multi-word expressions
    ID, UPOS = data_df.ID, data_df.UPOS
    range_ids = ID[ID.str.contains('-', na=False, regex=False)]

    # split range IDs (7-8 -> 7 8) and compute interval length ((8 - 7) + 1 = 2)
    range_id_intervals = range_ids.str.split('-', expand=True).astype(int)
    range_id_len = range_id_intervals[1] - range_id_intervals[0] + 1

    mask = np.zeros(len(data_df), dtype=bool)
    multiword_mask = np.zeros(len(data_df), dtype=bool)

    for range_len, range_idx in range_id_len.groupby(range_id_len):  # group range IDs by range length
        mask.fill(False)
        for start_range in range_idx.index+1: mask[start_range:start_range+range_len] = True

        upos = UPOS[mask]
        multiword_mask |= mask

        # concatenate UPOS tags of range ID tokens (7-8 -> ADP+DET -> UPOS of token 7 and 8)
        upos_concat = upos[::range_len].str.cat([upos[start_idx::range_len].values
                                       for start_idx in range(1, range_len)], sep='+')

        upos_concat = upos_concat.map(multiword_upos_map) if multiword_upos_map is not None else upos_concat
        UPOS.loc[range_idx.index] = upos_concat.values

    # discard multi-word tokens that belong to range IDs (7-8 -> discard token 7 and 8)
    data_df = data_df[~multiword_mask]
    return data_df

def count_sents(data_df):
    """
    Retrieves number of sentences and number of tokens from a DataFrame.
    """
    num_tokens = data_df.Token.count()  # counts non-NaN rows
    num_sents = len(data_df) - num_tokens  # counts NaN rows

    # if last NaN is missing then count one more for the last sentence
    if NOT_SENT_BOUNDARY(data_df.Token.iloc[-1]):
        num_sents += 1

    return num_sents, num_tokens

def dataframe_to_sents(data_df):
    """
    Converts a DataFrame into a list of sentences, where in turn each
    sentence is a list of tokens. A token is represented as a namedtuple.
    """
    num_sents, _ = count_sents(data_df)
    sents = [[] for idx in range(num_sents)]
    sent_idx = 0

    for row in data_df.itertuples(index=False, name='Token'):
        if NOT_SENT_BOUNDARY(row.Token):
            sents[sent_idx].append(row)
        else:
            sent_idx += 1

    return sents

def sents_to_dataframe(sents):
    """
    Converts a list of sentences, composed of lists of tokens, into a DataFrame.
    """
    num_cols = len(sents[0][0])
    sent_boundary_token = (SENT_BOUNDARY,) * num_cols
    data_df = pd.DataFrame(_sents_to_dataframe(sents, sent_boundary_token))

    return data_df

def _sents_to_dataframe(sents, sent_boundary_token):
    for sent in sents:
        for token in sent:
            yield token
        yield sent_boundary_token

def train_test_split(data_df, data_split):
    """
    Splits the input data into train, dev and tests sets. The order
    of the sentences is randomized before making the split.

    :param data_df: A DataFrame containing the input data to split on.
    :param data_split: A dictionary containing the proportions of the
    train, dev and test sets as a percentage, which must sum to one.
    """
    assert sum(data_split.values()) == 1.0, 'Total proportion of train, dev and test splits must sum to 1.0'

    sents = dataframe_to_sents(data_df)
    num_sents = len(sents)

    # randomize the ordering of the sentences before splitting
    random.shuffle(sents)

    # compute the integer offsets to split on
    train_split = int(data_split['train'] * num_sents)
    dev_split = int((data_split['train'] + data_split['dev']) * num_sents)

    # split data into train, dev and test sets
    train_sents = sents[:train_split]
    dev_sents = sents[train_split:dev_split]
    test_sents = sents[dev_split:]

    data_splitted = {
        'train': sents_to_dataframe(train_sents),
        'dev': sents_to_dataframe(dev_sents),
        'test': sents_to_dataframe(test_sents),
    }

    return data_splitted

def write_data(data_df, dataset_out_path, output_separator, output_columns):
    """
    Writes the data to a file in a format similar in scope to CoNLL-2003.
    The output file will contain one token per line, where empty lines denote
    sentence boundaries. Token attributes are joined by an output separator.
    """
    # create output dataset directory if it doesn't yet exist
    dataset_out_dir = dataset_out_path.parent
    if not dataset_out_dir.exists():
        dataset_out_dir.mkdir(parents=True)

    # keep the columns (and column order) as defined in output_columns
    output_df = data_df[output_columns]

    # concatenate the Token column with all other columns via an output separator
    output_rows = output_df.Token.str.cat(others=output_df.drop(columns='Token'),
                                          sep=output_separator)

    # separate tokens by newline characters where NaN rows become empty lines
    output_text = output_rows.str.cat(sep='\n', na_rep='')
    dataset_out_path.write_text(output_text)

def replace_values(values, from_column, to_column=None):
    """
    Replaces Given a dictionary

    :param values: A dictionary
    :param from_column: A DataFrame column 
    :param to_column: A DataFrame column 
    """
    # if only one dataframe column is passed then replace values on that column
    if to_column is None:
        to_column = from_column

    # locates all column elements on values.keys() and replaces them by their values
    values_dtype = CategoricalDtype(categories=values.keys())
    mapped_values = from_column.astype(values_dtype).map(values)

    # writes the column elements that were replaced on to_column
    to_column.where(pd.isna(mapped_values), mapped_values, inplace=True)


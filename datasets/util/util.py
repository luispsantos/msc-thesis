import numpy as np
import pandas as pd
from pathlib import Path
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

def read_conllu(dataset_in_path, remove_multiwords=True, keep_columns=['FORM', 'UPOS']):
    """
    Reads a CoNLL-U file into a DataFrame. Optionally removes multi-word tokens.
    The CoNLL-U format specification can be found at http://universaldependencies.org/format.html.
    """
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    data_df = pd.read_csv(dataset_in_path, sep='\t', names=column_names, skip_blank_lines=False, comment='#')

    # remove multi-word tokens by skipping over all range IDs
    if remove_multiwords:
        data_df = data_df[data_df.ID.isna() | data_df.ID.str.isdecimal()]

    # keep only a few columns of the dataset and discard the rest
    data_df = data_df[keep_columns]
    data_df.rename(columns={'FORM': 'Token', 'UPOS': 'POS'}, inplace=True)

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

def write_data(data_df, dataset_out_path, output_separator):
    """
    Writes the data to a file in a format similar in scope to CoNLL-2003.
    The output file will contain one token per line, where empty lines denote
    sentence boundaries. Token attributes are joined by an output separator.
    """
    # concatenate the Token column with all other columns via an output separator
    output_rows = data_df.Token.str.cat(others=data_df.drop(columns='Token'),
                                        sep=output_separator)

    # separate tokens by newline characters where NaN rows become empty lines
    output_text = output_rows.str.cat(sep='\n', na_rep='')
    dataset_out_path.write_text(output_text)


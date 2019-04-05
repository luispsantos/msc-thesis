from .util import IS_SENT_BOUNDARY
import numpy as np
import pandas as pd
import random

# fix the random seed for reproducibility
random.seed(123)

def _dataframe_to_sents(data_df):
    """
    Converts a DataFrame of tokens into a list of sentences.
    """
    is_sent_boundary = IS_SENT_BOUNDARY(data_df.Token)
    sent_end_idxs = np.where(is_sent_boundary)[0]

    # split the DataFrame at every sentence start
    sent_start_idxs = sent_end_idxs + 1
    sents = np.vsplit(data_df.values, sent_start_idxs)

    return sents

def _sents_to_dataframe(sents, columns):
    """
    Converts a list of sentences into a DataFrame of tokens.
    """
    data_df = pd.DataFrame(np.vstack(sents), columns=columns)

    return data_df

def train_test_split(data_df, data_split=None):
    """
    Splits the input data into train, dev and tests sets. The order
    of the sentences is randomized before making the split.

    :param data_df: A DataFrame containing the input data to split on.
    :param data_split: A dictionary containing the proportions of the
    train, dev and test sets as a percentage, which must sum to one.
    """
    if data_split is None:
        data_split = {'train': 0.75, 'dev': 0.1, 'test': 0.15}

    assert sum(data_split.values()) == 1.0, 'Total proportion of train, dev ' \
                                            'and test splits must sum to 1.0'

    sents = _dataframe_to_sents(data_df)
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

    train_test_dfs = {
        'train': _sents_to_dataframe(train_sents, data_df.columns),
        'dev':   _sents_to_dataframe(dev_sents, data_df.columns),
        'test':  _sents_to_dataframe(test_sents, data_df.columns),
    }

    return train_test_dfs


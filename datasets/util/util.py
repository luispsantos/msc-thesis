import numpy as np
import pandas as pd

SENT_BOUNDARY = np.nan
IS_SENT_BOUNDARY = lambda val: pd.isna(val)
NOT_SENT_BOUNDARY = lambda val: pd.notna(val)

def read_data(token_generator, token_re):
    # create a Series of raw tokens by means of a token generator on the input dataset file
    raw_tokens = pd.Series(token_generator)

    # make sure the token regex matches the whole token
    assert raw_tokens.str.match(token_re).all(), 'Token regex failed to match at least one token'

    # convert a Series of raw tokens into a DataFrame of cleaned-up strings
    # where each capture group names in the regex are used as column names
    data_df = raw_tokens.str.extract(token_re)

    return data_df

def dataframe_to_sents(data_df):

    num_sents = data_df.Token.isna().sum()
    sents = [[] for idx in range(num_sents)]
    sent_idx = 0

    for row in data_df.itertuples(index=False, name='Token'):
        if NOT_SENT_BOUNDARY(row.Token):
            sents[sent_idx].append(row)
        else:
            sent_idx += 1

    return sents

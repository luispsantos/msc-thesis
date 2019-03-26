from pathlib import Path
import numpy as np
import pandas as pd
import csv

SENT_BOUNDARY = np.nan
IS_SENT_BOUNDARY = lambda val: pd.isna(val)
NOT_SENT_BOUNDARY = lambda val: pd.notna(val)

def read_data(sents, token_re):
    """
    Creates a DataFrame containing the tokens as the rows and token information
    such as the POS/NER tag or lemma of the token as columns. To indicate sentence
    boundaries, the special token SENT_BOUNDARY will be added between sentences.

    :param sents: an iterable that yields sentences from the original dataset file.
    :param token_re: a regex to extract the token attributes from the raw token by
    means of capture groups. Each capture group in the regex becomes a new column.
    """
    num_groups, group_names = token_re.groups, list(token_re.groupindex)
    sent_boundary_token = (SENT_BOUNDARY,) * num_groups
    tokens = []

    for sent in sents:
        # split each sentence into tokens and extract the regex's capture groups
        for raw_token in sent.split():
            token_match = token_re.match(raw_token)
            assert token_match, f'Token regex failed to match {raw_token}'
            tokens.append(token_match.groups())

        # append the sentence boundary token between sentences
        tokens.append(sent_boundary_token)

    data_df = pd.DataFrame.from_records(tokens, columns=group_names)

    return data_df

def read_conll(data_in_path, sep, column_names):
    """Reads a CoNLL file into a DataFrame."""
    data_df = pd.read_csv(data_in_path, sep=sep, names=column_names,
                          skip_blank_lines=False, quoting=csv.QUOTE_NONE)

    return data_df

def read_conllu(data_in_path, keep_columns=['ID', 'FORM', 'UPOS']):
    """
    Reads a CoNLL-U file into a DataFrame. Optionally discards some of the columns.
    The CoNLL-U format specification can be found at http://universaldependencies.org/format.html.
    """
    column_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
    data_df = pd.read_csv(data_in_path, sep='\t', names=column_names,
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
    # compute tokens with range IDs (e.g. 7-8) which define multi-word tokens
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

def write_data(data_df, data_out_path, output_columns, output_separator='\t'):
    """
    Writes the data to a file in a format similar in scope to CoNLL-2003.
    The output file will contain one token per line, where empty lines denote
    sentence boundaries. Token attributes are joined by an output separator.
    This function accepts one of two calling conventions: either data_df is a single
    DataFrame or a dict of DataFrames, in which case the keys are used as filenames.

    :param data_df: A DataFrame containing the output data or a dict of DataFrames.
    :param data_out_path: a file path to write the data if a single DataFrame.
    was given, otherwise on a dict of DataFrames it should be a directory path.
    :param output_columns: list of column names to retain on the output file.
    :param output_separator: a delimiter to separate columns (e.g. ' ', '\t').
    """
    if isinstance(data_df, dict):
        train_test_dfs, dataset_out_dir = data_df, data_out_path
        
        for dataset_type, data_df in train_test_dfs.items():
            data_out_path = dataset_out_dir / (dataset_type + '.txt')
            write_data(data_df, data_out_path, output_columns, output_separator)
    else:        
        # create output dataset directory if it doesn't yet exist
        data_out_path.parent.mkdir(exist_ok=True)

        # keep the columns (and column order) as defined in output_columns
        output_df = data_df[output_columns]

        # concatenate the Token column with all other columns via an output separator
        output_rows = output_df.Token.str.cat(others=output_df.drop(columns='Token'),
                                          sep=output_separator)

        # separate tokens by newline characters where NaN rows become empty lines
        output_text = output_rows.str.cat(sep='\n', na_rep='')
        data_out_path.write_text(output_text)


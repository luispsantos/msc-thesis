from pathlib import Path
from .util import SENT_BOUNDARY, dump_yaml
from .stats import compute_stats
import numpy as np
import pandas as pd
import csv

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

def dataset_with_splits(process_dataset, data_in_files, dataset_in_dir, dataset_out_dir, output_columns):
    """
    Processes a dataset with pre-made train, dev and test splits. The splits are maintained
    and the data is not shuffled. The function process_dataset is thus applied to each split
    independently. The argument data_in_files can be a dict of input files or a str.format instance.
    """
    # if data_in_files is a str.format then call it for train, dev and test sets
    if not isinstance(data_in_files, dict):
        data_in_files = {dataset_type: data_in_files(dataset_type=dataset_type)
                         for dataset_type in ['train', 'dev', 'test']}
    
    train_test_dfs = {}
    for dataset_type, data_in_file in data_in_files.items():
        data_in_path = dataset_in_dir / data_in_file

        data_df = process_dataset(data_in_path)
        train_test_dfs[dataset_type] = data_df

    # write data to disk
    write_data(train_test_dfs, dataset_out_dir, output_columns)

def write_data(train_test_dfs, dataset_out_dir, output_columns, output_separator='\t'):
    """
    Writes the data to a file in a format similar in scope to CoNLL-2003.
    The output file will contain one token per line, where empty lines denote
    sentence boundaries. Token attributes are joined by an output separator.

    :param train_test_df: A dict of DataFrames containing the output data in
    the form of a train, dev and test sets. The keys will be used as filenames.
    :param dataset_out_dir: a directory to write the output data files.
    :param output_columns: list of column names to retain on the output file.
    :param output_separator: a delimiter to separate columns (e.g. ' ', '\t').
    """
    # create output dataset directory if it does not exist
    dataset_out_dir.mkdir(exist_ok=True)

    dataset_stats = {}

    # compute number of unique tokens - vocabulary size for the dataset
    dataset_stats['vocab_size'] = len({token for data_df in train_test_dfs.values()
                                       for token in data_df.Token.dropna()})

    for dataset_type, data_df in train_test_dfs.items():
        data_out_path = dataset_out_dir / (dataset_type + '.txt')

        # compute some dataset statistics
        dataset_stats[dataset_type] = compute_stats(data_df)

        # keep the columns (and column order) as defined in output_columns
        output_df = data_df[output_columns]

        # concatenate the Token column with all other columns via an output separator
        output_rows = output_df.Token.str.cat(others=output_df.drop(columns='Token'),
                                          sep=output_separator)

        # separate tokens by newline characters where NaN rows become empty lines
        output_text = output_rows.str.cat(sep='\n', na_rep='')
        data_out_path.write_text(output_text)

    # write dataset statistics to a YAML file
    dump_yaml('stats.yml', dataset_stats)


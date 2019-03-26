from pathlib import Path
from pandas.api.types import CategoricalDtype
from .dataset_io import IS_SENT_BOUNDARY
import numpy as np
import pandas as pd
import yaml

def load_yaml(yaml_file):
    """Loads a specified YAML file."""
    with open(yaml_file, 'r') as f:
        yaml_parsed = yaml.safe_load(f)

    return yaml_parsed

def replace_values(values, from_column, to_column=None):
    """
    Finds all of values.keys() on from_column and replaces these by their
    corresponding values on to_column. from_column and to_column can be identical.

    :param values: A dictionary of values to replace on to_column.
    :param from_column: A DataFrame column to search for the value keys.
    :param to_column: A DataFrame column to replace by the matching values.
    """
    # if only one dataframe column is passed then replace values on that column
    if to_column is None:
        to_column = from_column

    # locates all column elements on values.keys() and replaces them by their values
    values_dtype = CategoricalDtype(categories=values.keys())
    mapped_values = from_column.astype(values_dtype).map(values)

    # writes the column elements that were replaced on to_column
    to_column.where(pd.isna(mapped_values), mapped_values, inplace=True)

def add_bio_encoding(ner):
    """Converts a NER column without a NER scheme (e.g. O LOC LOC O) to BIO."""
    # add I- prefix to all tags except O tags
    ner.where(ner == 'O', 'I-' + ner, inplace=True)
    iob_to_bio(ner)

def iob_to_bio(ner):
    """
    Converts from IOB format (e.g. O I-LOC I-LOC) to BIO format (e.g. O B-LOC I-LOC).
    A tag with the prefix I- takes the prefix B- in the case of one of the following:
    1 - Previous tag is a sentence boundary (i.e., current tag is a sentence start).
    2 - Previous tag is tagged as O (i.e., previous tag is outside a named entity).
    3 - Previous tag is different from current tag (i.e., different named entities).
    """
    prev_ner = ner.shift()
    ner.mask((ner.str[:2] == 'I-') & (IS_SENT_BOUNDARY(prev_ner) |
                                     (prev_ner == 'O') |
                                     (prev_ner.str[2:] != ner.str[2:])),
                                     ner.str.replace('I-', 'B-'), inplace=True)


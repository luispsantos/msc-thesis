from .util import NOT_SENT_BOUNDARY
import pandas as pd

def count_sents(data_df):
    """
    Retrieves number of sentences and number of tokens from a DataFrame.
    """
    num_tokens = int(data_df.Token.count())  # counts non-NaN rows
    num_sents = len(data_df) - num_tokens    # counts NaN rows

    # if last NaN is missing then count one more for the last sentence
    if NOT_SENT_BOUNDARY(data_df.Token.iloc[-1]):
        num_sents += 1

    return num_sents, num_tokens

def compute_stats(data_df):
    """
    Computes some statistics on a dataset (e.g. sent/token counts, POS/NER tagset).
    """
    stats = {}

    num_sents, num_tokens = count_sents(data_df)
    stats['sents'], stats['tokens'] = num_sents, num_tokens

    if 'UPOS' in data_df.columns:
        # count number of syntactic words (e.g. do -> 1 token & 2 words)
        stats['words'] = int((data_df.UPOS.str.count('\+') + 1).sum())

        upos_counts = data_df.UPOS.value_counts()
        stats['POS'] = upos_counts.sort_index().to_dict()

    if 'NER' in data_df.columns:
        # count number of entities (each B-X is a count of one for entity X)
        ent_start_mask = data_df.NER.str.startswith('B-', na=False)
        ents = data_df.loc[ent_start_mask, 'NER'].str[len('B-'):]

        ner_counts = ents.value_counts()
        stats['NER'] = ner_counts.sort_index().to_dict()

    return stats

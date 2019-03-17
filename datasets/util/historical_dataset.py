from pathlib import Path
import numpy as np
import pandas as pd
import multiprocessing
import functools

def _process_text_file(preprocess_text, text_to_dataframe, text_file):
    text = text_file.read_text()
    text = preprocess_text(text)
    text_df = text_to_dataframe(text)
    return text_df

def read_text_files(dataset_in_dir, preprocess_text, text_to_dataframe):
    """
    Recursively finds all text files inside a directory and reads those in parallel to a
    DataFrame. Each token will be indexed by a hierarchical index of the text ID and token ID.
    This allows to compute statistics such as the number of tokens and sentences per text file.

    :param dataset_in_dir: the directory that contains the corpus texts inside .txt files.
    :param preprocess_text: a function that corrects problems in the data at the string level.
    :param text_to_dataframe: a function that receives the corpus text and outputs a DataFrame.
    """
    text_files = sorted(dataset_in_dir.rglob('*.txt'))
    filenames = [text_file.stem for text_file in text_files]

    # set constant parameters for functions preprocess_text and text_to_dataframe
    process_text_file = functools.partial(_process_text_file, preprocess_text,
                                          text_to_dataframe)
    with multiprocessing.Pool() as pool:
        text_dfs = pool.map(process_text_file, text_files)
        data_df = pd.concat(text_dfs, keys=filenames, names=['TextId', 'TokenId'])

    return data_df

def sent_tokenize(data_df, token_col, spacy_model='pt_core_news_sm'):
    """
    Tokenizes text into sentences using spaCy. It preserves the pre-tokenization of the data
    (i.e. lets spaCy tokenize sentences but not tokenize words). In spaCy models, the role
    of sentence tokenization is performed jointly with the DependencyParser, and so the parser
    must run for all texts in the corpus. Documents are processed in parallel via OpenMP.

    :param data_df: a DataFrame with a MultiIndex containing the input data.
    :param token_col: a column from the DataFrame containing the input tokens.
    :param spacy_model: the name of a spaCy model to use for sentence tokenization.
    """
    import spacy
    from spacy.tokens import Doc

    # load spaCy model with the dependency parser
    nlp = spacy.load(spacy_model, disable=['tagger', 'ner'])

    # create a list of tokens for each document/text file and create a spaCy Doc
    # with the same tokenization (i.e. do not let spaCy tokenize the text for us).
    tokens_list = token_col.groupby(['TextId']).apply(list)
    docs = [Doc(nlp.vocab, words=words) for words in tokens_list]

    # process the texts in parallel and save all computed sentence boundaries
    sent_end_idxs, doc_start_idx = [], 0
    for doc in nlp.parser.pipe(docs, batch_size=25, n_threads=-1):
        assert doc.is_parsed
        sent_end_idxs += [doc_start_idx+sent.end for sent in doc.sents]
        doc_start_idx += len(doc)

    # convert the data values and index to NumPy arrays
    data_values, index_values = data_df.values, data_df.index.to_frame().values
    
    # insert the sentence boundary token (NaN) between sentences
    # as for the index repeat the index value before the sentence end
    values_with_sents = np.insert(data_values, sent_end_idxs, np.nan, axis=0)
    index_with_sents = np.insert(index_values, sent_end_idxs,
                                 index_values[np.array(sent_end_idxs)-1], axis=0)

    # create a new DataFrame containing the tokens and computed sentence boundaries
    df_with_sents = pd.DataFrame(values_with_sents, columns=data_df.columns)
    df_with_sents.index = pd.MultiIndex.from_frame(pd.DataFrame(index_with_sents),
                                                   names=data_df.index.names)

    return df_with_sents


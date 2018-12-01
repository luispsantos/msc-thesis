import pandas as pd
from pathlib import Path
import argparse
from util.dataset import Dataset

import spacy
from spacy.tokens import Doc
from spacy.gold import GoldParse, iob_to_biluo
from spacy.scorer import Scorer

parser = argparse.ArgumentParser(description='Apply the Spacy Tagger and EntityRecognizer on a dataset and evaluate ' \
                                             'according to the gold annotations (Accuracy for PoS, F1 score for NER).')
parser.add_argument('dataset_path', help='Path to a dataset containing a data folder')

args = parser.parse_args()
dataset_path = Path(args.dataset_path)

# load dataset
dataset = Dataset(dataset_path)

# load spacy model
nlp = spacy.load('pt_core_news_sm')

for dataset_type in ['train', 'dev', 'test']:
    data_df = dataset.data[dataset_type]

    # store a boolean array specifying when sentences start
    data_df['is_sent_start'] = data_df.Token.isna().shift(1)
    data_df.loc[0, 'is_sent_start'] = True

    # remove NaN values
    data_df.dropna(inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    # create spacy Doc with pre-tokenized text
    # this construction doesn't call pipeline components
    doc = Doc(nlp.vocab, list(data_df.Token))

    # store sentence start flag as token attributes
    # this information will be used by the EntityRecognizer
    for token, flag in zip(doc, data_df.is_sent_start):
        token.is_sent_start = flag
    
    tags_gold, ents_gold = None, None

    if 'POS' in data_df.columns:
        # apply the Tagger to assign PoS tags
        nlp.tagger(doc)
        data_df['spacy_POS'] = pd.Series(token.pos_ for token in doc)
        tags_gold = list(data_df.POS)

        # overwrite fine-grained PoS tag with the coarse-grained tag from UD
        # this is necessary since the Scorer() evaluates on fine-grained tags
        for token in doc:
            token.tag_ = token.pos_

    if 'NER' in data_df.columns:
        # apply the EntityRecognizer to assign NER tags
        nlp.entity(doc)
        data_df['spacy_NER'] = pd.Series('{}-{}'.format(token.ent_iob_, token.ent_type_)
                                         if token.ent_iob_ != 'O' else 'O' for token in doc)

        iob_ents = list(data_df.NER)
        ents_gold = iob_to_biluo(iob_ents)

    # create a GoldParse object with the true annotations
    gold = GoldParse(doc, tags=tags_gold, entities=ents_gold)

    # compute scoring metrics
    scorer = Scorer()
    scorer.score(doc, gold)

    if 'POS' in data_df.columns:
        print('{} acc: {:.4f}'.format(dataset_type.capitalize(), scorer.tags_acc))

    if 'NER' in data_df.columns:
        print('{} - Prec: {:.4f}, Rec: {:.4f}, F1: {:4f}'.format(dataset_type.capitalize(), scorer.ents_p, scorer.ents_r, scorer.ents_f))


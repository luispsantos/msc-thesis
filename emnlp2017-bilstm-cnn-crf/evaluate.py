import pandas as pd
from pathlib import Path
from io import StringIO
from seqeval.metrics import accuracy_score, classification_report

class Evaluate:

    def __init__(self):
        self.pos_metrics, self.pos_tags_f1 = {}, {}
        self.ner_metrics, self.ent_type_f1 = {}, {}

    def eval_pos(self, dataset_name, corr_labels, pred_labels, train_data, test_data):
        # compute flat list of all tokens and POS tags on training set
        train_tokens = [token for sent in train_data for token in sent['raw_tokens']]
        train_labels = [label for sent in train_data for label in sent['POS']]

        # find all tokens on test set that do not appear on the training set
        train_tokens_unique = set(train_tokens)
        unseen_mask = [token not in train_tokens_unique for sent in test_data for token in sent['raw_tokens']]

        # compute number of distinct POS tags per train token
        train_df = pd.DataFrame({'Token': train_tokens, 'POS': train_labels})
        tags_per_token = train_df.groupby('Token')['POS'].nunique()

        # find all train tokens which can have distinct POS tags
        ambiguous_tokens = set(tags_per_token[tags_per_token > 1].index)
        ambiguous_mask = [token in ambiguous_tokens for sent in test_data for token in sent['raw_tokens']]

        # flatten a 2D list of sent and token labels into a 1D list of labels
        corr_labels = [label for sent_labels in corr_labels for label in sent_labels]
        pred_labels = [label for sent_labels in pred_labels for label in sent_labels]

        # condition tokens based on whether they were seen on the training set
        unseen_corr_labels = [label for label, is_unseen in zip(corr_labels, unseen_mask) if is_unseen]
        unseen_pred_labels = [label for label, is_unseen in zip(pred_labels, unseen_mask) if is_unseen]

        # condition tokens based on whether they are ambiguous (can have multiple POS tags)
        ambiguous_corr_labels = [label for label, is_ambiguous in zip(corr_labels, ambiguous_mask) if is_ambiguous]
        ambiguous_pred_labels = [label for label, is_ambiguous in zip(pred_labels, ambiguous_mask) if is_ambiguous]

        # compute overall, unseen and ambiguous tokens accuracy
        overall_acc = round(accuracy_score(corr_labels, pred_labels) * 100, 2)
        unseen_acc = round(accuracy_score(unseen_corr_labels, unseen_pred_labels) * 100, 2)
        ambiguous_acc = round(accuracy_score(ambiguous_corr_labels, ambiguous_pred_labels) * 100, 2)

        # insert B- prefix on all POS tags and remove padding token
        corr_labels_bio = ['B-'+label for label in corr_labels if label != 'O']
        pred_labels_bio = ['B-'+label for label in pred_labels if label != 'O']

        # compute Prec, Rec and F1 metrics per POS tag
        metrics = classification_report(corr_labels_bio, pred_labels_bio, digits=4)
        metrics = pd.read_csv(StringIO(metrics), sep=' {2,}', engine='python') * 100

        # sort the labels alphabetically and rename columns
        metrics.sort_index(inplace=True)
        metrics.rename(columns={'precision': 'Prec', 'recall': 'Rec', 'f1-score': 'F1'}, inplace=True)

        tok_avg = metrics.loc['avg / total'].drop('support')
        self.pos_metrics[dataset_name] = {'Overall': overall_acc, 'Unseen': unseen_acc,
                                    'Ambiguous': ambiguous_acc, **tok_avg}

        self.pos_tags_f1[dataset_name] = metrics['F1'].drop('avg / total')

    def eval_ner(self, dataset_name, corr_labels, pred_labels, train_data, test_data):
        # compute entity-level metrics
        metrics = classification_report(corr_labels, pred_labels, digits=4)
        metrics = pd.read_csv(StringIO(metrics), sep=' {2,}', engine='python') * 100

        # sort the labels alphabetically and rename columns
        metrics.sort_index(inplace=True)
        metrics.rename(columns={'precision': 'Prec', 'recall': 'Rec', 'f1-score': 'F1'}, inplace=True)

        # append the prefix B- to all tags in order to compute token-level metrics
        corr_labels_t = [['B'+ent[1:] if ent[0] in ('B', 'I') else 'B-'+ent for ent in sent] for sent in corr_labels]
        pred_labels_t = [['B'+ent[1:] if ent[0] in ('B', 'I') else 'B-'+ent for ent in sent] for sent in pred_labels]

        #corr_labels_t = [['B'+ent[1:] if ent[0] == 'I' else ent for ent in sent] for sent in corr_labels]
        #pred_labels_t = [['B'+ent[1:] if ent[0] == 'I' else ent for ent in sent] for sent in pred_labels]

        # compute token-level metrics
        metrics_t = classification_report(corr_labels_t, pred_labels_t, digits=4)
        metrics_t = pd.read_csv(StringIO(metrics_t), sep=' {2,}', engine='python') * 100

        # sort the labels alphabetically and rename columns
        metrics_t.sort_index(inplace=True)
        metrics_t.rename(columns={'precision': 'Prec', 'recall': 'Rec', 'f1-score': 'F1'}, inplace=True)

        # obtain overall Prec, Rec and F1 for entity- and token-level
        ent_avg = metrics.loc['avg / total'].drop('support')
        tok_avg = metrics_t.loc['avg / total'].drop('support')

        metrics_avg = pd.concat([ent_avg, tok_avg], keys=['Entity', 'Token'])
        self.ner_metrics[dataset_name] = metrics_avg

        ent_f1 = metrics['F1'].drop('avg / total')
        tok_f1 = metrics_t['F1'].drop('avg / total')

        metrics_f1 = pd.concat([ent_f1, tok_f1], keys=['Entity', 'Token'])
        self.ent_type_f1[dataset_name] = metrics_f1

    def eval(self, dataset_name, task, corr_labels, pred_labels, train_data, test_data):
        if task == 'POS':
            self.eval_pos(dataset_name, corr_labels, pred_labels, train_data, test_data)
        elif task == 'NER':
            self.eval_ner(dataset_name, corr_labels, pred_labels, train_data, test_data)

    def write_tables(self, tables_dir):
        # make sure that tables directory exists
        tables_dir.mkdir(exist_ok=True)

        pos_metrics = pd.DataFrame.from_dict(self.pos_metrics, orient='index')
        pos_tags_f1 = pd.DataFrame.from_dict(self.pos_tags_f1, orient='columns')

        ner_metrics = pd.DataFrame.from_dict(self.ner_metrics, orient='index')
        ent_type_f1 = pd.DataFrame.from_dict(self.ent_type_f1, orient='index')

        pos_metrics.to_latex(tables_dir / 'pos_metrics.tex', na_rep='-')
        pos_tags_f1.to_latex(tables_dir / 'pos_tags_f1.tex', na_rep='-')

        ner_metrics.to_latex(tables_dir / 'ner_metrics.tex', na_rep='-')
        ent_type_f1.to_latex(tables_dir / 'ent_type_f1.tex', na_rep='-')

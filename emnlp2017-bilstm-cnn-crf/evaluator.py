import pandas as pd
from pathlib import Path
from io import StringIO
from seqeval.metrics import accuracy_score, performance_measure, classification_report

class Evaluator:

    def __init__(self):
        self.pos_metrics, self.pos_tags_f1 = {}, {}
        self.ner_metrics, self.ent_type_f1 = {}, {}

    def eval_pos(self, dataset_id, corr_labels, pred_labels, train_data, test_data):
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

        acc_metrics = pd.Series({'Overall': overall_acc, 'Unseen': unseen_acc, 'Ambiguous': ambiguous_acc})
        tok_avg = metrics.loc['micro avg'].drop('support')

        metrics_avg = pd.concat([acc_metrics, tok_avg], keys=['Accuracy', 'NaN'])
        self.pos_metrics[dataset_id] = metrics_avg

        self.pos_tags_f1[dataset_id] = metrics['F1'].drop(['micro avg', 'macro avg'])

    def eval_ner(self, dataset_id, corr_labels, pred_labels, train_data, test_data):
        # compute entity-level metrics
        metrics = classification_report(corr_labels, pred_labels, digits=4)
        metrics = pd.read_csv(StringIO(metrics), sep=' {2,}', engine='python') * 100

        # sort the labels alphabetically and rename columns
        metrics.sort_index(inplace=True)
        metrics.rename(columns={'precision': 'Prec', 'recall': 'Rec', 'f1-score': 'F1'}, inplace=True)

        # append the prefix B- to all tags in order to compute token-level metrics
        corr_labels_t = [['B'+ent[1:] if ent[0] in ('B', 'I') else 'B-'+ent for ent in sent] for sent in corr_labels]
        pred_labels_t = [['B'+ent[1:] if ent[0] in ('B', 'I') else 'B-'+ent for ent in sent] for sent in pred_labels]

        # compute token-level metrics
        metrics_t = classification_report(corr_labels_t, pred_labels_t, digits=4)
        metrics_t = pd.read_csv(StringIO(metrics_t), sep=' {2,}', engine='python') * 100

        # sort the labels alphabetically and rename columns
        metrics_t.sort_index(inplace=True)
        metrics_t.rename(columns={'precision': 'Prec', 'recall': 'Rec', 'f1-score': 'F1'}, inplace=True)

        # compute performance metrics
        perf = performance_measure(corr_labels, pred_labels)
        tp, tn, fp, fn = perf['TP'], perf['TN'], perf['FP'], perf['FN']

        # compute entity- and token-level accuracy
        ent_acc = round((tp + tn) / (tp + tn + fp + fn) * 100, 2)
        tok_acc = round(accuracy_score(corr_labels, pred_labels) * 100, 2)

        # obtain overall Prec, Rec and F1 for entity- and token-level
        ent_avg = metrics.loc['micro avg'].drop('support')
        tok_avg = metrics_t.loc['micro avg'].drop('support')

        ent_avg = pd.concat([pd.Series({'Acc': ent_acc}), ent_avg])
        tok_avg = pd.concat([pd.Series({'Acc': tok_acc}), tok_avg])

        metrics_avg = pd.concat([ent_avg, tok_avg], keys=['Entity Spans', 'Tokens'])
        self.ner_metrics[dataset_id] = metrics_avg

        # obtain F1 score at the entity- and token-level per entity type
        ent_f1 = metrics['F1'].drop(['micro avg', 'macro avg'])
        tok_f1 = metrics_t['F1'].drop(['O', 'micro avg', 'macro avg'])

        metrics_f1 = pd.concat([ent_f1, tok_f1], keys=['Entity Spans', 'Tokens'])
        self.ent_type_f1[dataset_id] = metrics_f1

    def eval(self, dataset_name, lang, task, corr_labels, pred_labels, train_data, test_data):
        dataset_id = (lang, dataset_name)
        if task == 'POS':
            self.eval_pos(dataset_id, corr_labels, pred_labels, train_data, test_data)
        elif task == 'NER':
            self.eval_ner(dataset_id, corr_labels, pred_labels, train_data, test_data)

    def write_tables(self, tables_dir):
        # make sure that the tables directory exists
        tables_dir.mkdir(exist_ok=True)
        print(f'Wrote evaluation tables to {tables_dir}/')

        self._to_latex(self.pos_metrics, tables_dir / 'pos_metrics.tex', True, False)
        self._to_latex(self.pos_tags_f1, tables_dir / 'pos_tags_f1.tex', False, True)

        self._to_latex(self.ner_metrics, tables_dir / 'ner_metrics.tex', True, False)
        self._to_latex(self.ent_type_f1, tables_dir / 'ent_type_f1.tex', False, False)

    def _to_latex(self, metrics, table_path, avg_row, transpose):
        # deal with the case of having no metrics to write
        if not metrics:
            return

        # create DataFrame and set index names
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.index.set_names(['Language', 'Dataset'], inplace=True)

        # show the Portuguese datasets at the top and Spanish datasets at the bottom
        metrics_df = metrics_df.reindex(['PT', 'ES'], level='Language')
        metrics_df.index.set_levels(['Portuguese', 'Spanish'], level='Language', inplace=True)

        # append a row that computes metric averages over all datasets per language
        if avg_row:
            metrics_avg = metrics_df.groupby('Language').mean().round(2) \
                                    .assign(Dataset='Average').set_index('Dataset', append=True)
            metrics_df = pd.concat([metrics_df, metrics_avg]).sort_values('Language')

        if transpose:
            metrics_df = metrics_df.transpose()
            metrics_df = metrics_df.sort_index()

        column_format = 'l' * metrics_df.index.nlevels + 'c' * len(metrics_df.columns)
        metrics_latex = metrics_df.to_latex(na_rep='-', column_format=column_format,
                                            multicolumn_format='c', multirow=True)

        # split latex tables into lines and replace \cline by \midrule
        lines = metrics_latex.splitlines()
        lines = [r'\midrule' if line.startswith(r'\cline') else line for line in lines]

        # add cmidrules for the header line
        toprule_idx = lines.index(r'\toprule')
        header_line = lines[toprule_idx+1]

        cmidrules, idx = [], 0
        for header in header_line.split('&'):
            if r'\multicolumn' in header:
                num_cols = int(header[header.index('{')+1])
                start_idx, end_idx = idx+1, idx+num_cols

                if 'NaN' not in header:
                    crule = f'\cmidrule(lr){{{start_idx}-{end_idx}}}'
                    cmidrules.append(crule)

            idx += num_cols if r'\multicolumn' in header else 1

        if cmidrules:
            lines[toprule_idx+1] = header_line.replace('NaN', '')
            lines.insert(toprule_idx+2, ' '.join(cmidrules))

        # aggregate index names and column names into the same table line
        midrule_idx = lines.index(r'\midrule')
        column_names, index_names = lines[midrule_idx-2:midrule_idx]

        col_names = [index_name if col_name.isspace() else col_name for col_name, index_name
                                        in zip(column_names.split('&'), index_names.split('&'))]
        col_names = '&'.join(col_names)

        if not transpose:
            lines.remove(column_names); lines.remove(index_names)
            lines.insert(lines.index(r'\midrule'), col_names)

        # write latex table to file
        table_path.write_text('\n'.join(lines))

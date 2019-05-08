from pathlib import Path
from util.util import load_yaml
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# use seaborn figure styles
sns.set()

def read_stats(data_dir):
    sent_tokens = {}
    pos_counts, ner_counts = {}, {}

    # iterate over the Portuguese and Spanish datasets and collect statistics
    for dataset_dir in chain(data_dir.glob('pt_*'), data_dir.glob('es_*')):
        # load config and stats YAML files
        dataset_config = load_yaml(dataset_dir / 'config.yml')
        dataset_stats = load_yaml(dataset_dir / 'stats.yml')

        # obtain dataset name and language from config file
        dataset_name = dataset_config['name']
        lang = dataset_config['lang']

        # iterate through stats of train, dev and test sets
        for dataset_type, stats in dataset_stats.items():
            dataset_id = (lang, dataset_name, dataset_type)
            sent_tokens[dataset_id] = {k: stats[k] for k in ('sents', 'tokens', 'words')
                                                   if k in stats}
            if 'POS' in stats:
                pos_counts[dataset_id] = stats['POS']

            if 'NER' in stats:
                ner_counts[dataset_id] = stats['NER']

    return sent_tokens, pos_counts, ner_counts

def create_dataframe(metrics):
    # create DataFrame and set index names
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.index.set_names(['Language', 'Dataset', 'Type'], inplace=True)

    # show the Portuguese datasets at the top and Spanish datasets at the bottom
    metrics_df = metrics_df.reindex(['PT', 'ES'], level='Language')
    metrics_df.index.set_levels(['Portuguese', 'Spanish'], level='Language', inplace=True)

    # sum metrics over the train, dev and test sets
    metrics_df = metrics_df.sum(level=['Language', 'Dataset'])
    metrics_df = metrics_df.astype('Int64').replace(0, np.nan)

    return metrics_df

def create_chart(metrics_df, plot_file, plot_args, legend_args,
                 subplots_args, column_wise, figplot, figsize):
    # group metrics per language
    lang_metrics = metrics_df.groupby(level='Language')
    num_datasets = lang_metrics.size().tolist()

    # create subplots for each language
    num_langs = len(lang_metrics)
    if column_wise:
        fig, axes = plt.subplots(nrows=num_langs, sharex=True,
                                figsize=figsize, gridspec_kw={
                                'height_ratios': num_datasets})
    else:
        fig, axes = plt.subplots(ncols=num_langs,
                                figsize=figsize, gridspec_kw={
                                'width_ratios': num_datasets})

    for ax, (lang, metrics) in zip(axes, lang_metrics):
        metrics = metrics.droplevel('Language')
        metrics.plot(ax=ax, **plot_args)

        ax.set_title(lang)
        ax.legend_.set_visible(False)

        ax.invert_yaxis()
        ax.yaxis.set_label_text('')

    if figplot:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_transform=ax.transAxes, **legend_args)
    else:
        axes[0].legend_.set_visible(True)
        axes[0].legend(**legend_args)

    plt.tight_layout()
    plt.subplots_adjust(**subplots_args)
    
    plt.savefig(plot_file)
    plt.close()

def sent_tokens_chart(metrics_df, plot_file):
    plot_args = {'kind': 'barh', 'logx': True}
    legend_args = {'loc': 'upper right'}
    subplots_args = {}

    create_chart(metrics_df, plot_file, plot_args, legend_args,
                 subplots_args, True, figplot=False, figsize=(8, 6))

def pos_counts_chart(metrics_df, plot_file):
    # convert the counts to percentages that sum to 1.0
    metrics_df = metrics_df.div(metrics_df.sum(axis=1), axis=0)

    plot_args = {'kind': 'barh', 'stacked': True}
    legend_args = {'loc': 'lower center', 'bbox_to_anchor': (1.2, -0.15),
                   'fontsize': 12}
    subplots_args = {'right': 0.75, 'hspace': 0.25}

    create_chart(metrics_df, plot_file, plot_args, legend_args,
                 subplots_args, True, figplot=True, figsize=(10, 6))

def ner_counts_chart(metrics_df, plot_file):
    # convert the counts to percentages that sum to 1.0
    metrics_df = metrics_df.div(metrics_df.sum(axis=1), axis=0)

    plot_args = {'kind': 'barh', 'stacked': True}
    legend_args = {'loc': 'lower center', 'bbox_to_anchor': (-0.25, -0.18),
                   'fontsize': 14, 'ncol': 3, 'shadow': True}
    subplots_args = {'bottom': 0.15}

    create_chart(metrics_df, plot_file, plot_args, legend_args,
                 subplots_args, False, figplot=True, figsize=(12, 6))

def to_latex(metrics_df, table_file):
    # create latex table from a DataFrame
    metrics_latex = metrics_df.to_latex(na_rep='-', multicolumn_format='c', multirow=True)

    # split latex table into lines and replace \cline by \midrule
    lines = metrics_latex.splitlines()
    lines = [r'\midrule' if line.startswith(r'\cline') else line for line in lines]

    # aggregate index names and column names into the same table line
    midrule_idx = lines.index(r'\midrule')
    column_names, index_names = lines[midrule_idx-2:midrule_idx]

    col_names = [index_name if col_name.isspace() else col_name for col_name, index_name
                                    in zip(column_names.split('&'), index_names.split('&'))]
    col_names = '&'.join(col_names)

    lines.remove(column_names); lines.remove(index_names)
    lines.insert(lines.index(r'\midrule'), col_names)

    # write latex table to file
    table_file.write_text('\n'.join(lines))

plots_dir = Path('plots')
cwd = Path.cwd()

# read dataset statistics
sent_tokens, pos_counts, ner_counts = read_stats(cwd)

# create DataFrames with stats for all datasets
sent_tokens = create_dataframe(sent_tokens)
pos_counts = create_dataframe(pos_counts)
ner_counts = create_dataframe(ner_counts)

sent_tokens_chart(sent_tokens, plots_dir / 'sent_tokens.png')
pos_counts_chart(pos_counts, plots_dir / 'pos_counts.png')
ner_counts_chart(ner_counts, plots_dir / 'ner_counts.png')

sent_tokens.columns = ['Sentences', 'Tokens', 'Words']
to_latex(sent_tokens, plots_dir / 'sent_tokens.tex')

to_latex(pos_counts, plots_dir / 'pos_counts.tex')
to_latex(ner_counts, plots_dir / 'ner_counts.tex')

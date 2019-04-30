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

def create_chart(metrics_df, plot_file, plot_args, legend_args, figsize):
    # group metrics per language
    lang_metrics = metrics_df.groupby(level='Language')
    num_datasets = lang_metrics.size().tolist()

    # create subplots for each language
    num_langs = len(lang_metrics)
    fig, axes = plt.subplots(nrows=num_langs, sharex=True,
                            figsize=figsize, gridspec_kw={
                            'height_ratios': num_datasets})

    for ax, (lang, metrics) in zip(axes, lang_metrics):
        metrics = metrics.droplevel('Language')
        metrics.plot(ax=ax, **plot_args)

        ax.set_title(lang)
        ax.legend_.set_visible(False)

        ax.invert_yaxis()
        ax.yaxis.set_label_text('')

    axes[0].legend_.set_visible(True)
    axes[0].legend(**legend_args)

    plt.tight_layout()
    plt.savefig(plot_file); plt.close()

def bar_chart(metrics_df, plot_file):
    plot_args = {'kind': 'barh', 'logx': True}
    legend_args = {'loc': 'upper right'}

    create_chart(metrics_df, plot_file, plot_args,
                 legend_args, figsize=(8, 6))

def stacked_bar_chart(metrics_df, plot_file, fontsize):
    # convert the counts to percentages that sum to 1.0
    metrics_df = metrics_df.div(metrics_df.sum(axis=1), axis=0)

    plot_args = {'kind': 'barh', 'stacked': True}
    legend_args = {'loc': 'center right', 'bbox_to_anchor': (1.4, 0.5),
                   'fontsize': fontsize}

    create_chart(metrics_df, plot_file, plot_args,
                 legend_args, figsize=(10, 6))

plots_dir = Path('plots')
cwd = Path.cwd()

# read dataset statistics
sent_tokens, pos_counts, ner_counts = read_stats(cwd)

# create DataFrames with stats for all datasets
sent_tokens = create_dataframe(sent_tokens)
pos_counts = create_dataframe(pos_counts)
ner_counts = create_dataframe(ner_counts)

bar_chart(sent_tokens, plots_dir / 'sent_tokens.png')
stacked_bar_chart(pos_counts, plots_dir / 'pos_counts.png', 8)
stacked_bar_chart(ner_counts, plots_dir / 'ner_counts.png', 12)

sent_tokens.to_latex(plots_dir / 'sent_tokens.tex', na_rep='-')
pos_counts.to_latex(plots_dir / 'pos_counts.tex', na_rep='-')
ner_counts.to_latex(plots_dir / 'ner_counts.tex', na_rep='-')

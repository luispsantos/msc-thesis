from pathlib import Path
from lxml import etree
from html import escape
import pandas as pd
import yaml
import re
import sys
import os

# change working dir into the directory containing the script
os.chdir(sys.path[0])

# importing from util/ directory
sys.path.insert(1, str(Path.cwd().parent / 'util'))
from rule_matcher import RuleMatcher, SequentialMatcher
from util import *

# read variables from the configuration file
with open('config.yml', 'r') as f:
    config = yaml.load(f)

dataset_in_dir, dataset_out_dir = Path(config['dataset_in_dir']), Path(config['dataset_out_dir'])
output_columns = config['output_columns']

# read dataset-specific rules
with open('rules.yml', 'r') as f:
    rules = yaml.load(f)

fix_contractions, fix_clitics = rules['fix_contractions'], rules['fix_clitics']
pos_tagset_ud_map, rules = rules['pos_tagset_ud_map'], rules['rules']

# load contractions for Portuguese
with open('contractions3.yml', 'r') as f:
    contractions = yaml.load(f)

# load clitic contractions for Portuguese
with open('clitics3.yml', 'r') as f:
    clitics = yaml.load(f)

# regex to extract Token, POS and NER tags from raw tokens
# capturing URLs uses a greedy version in order to match trailing slashes
# discards information about spaces before/after, lemma and token features
token_re = re.compile(r'(?:\\\*)?(?P<Token>http:.+|.+?)(?:\*\/)?(?:/.+)?(?:/(?P<POS>[A-Z]+)\d?)(?:#.+)?(?:\[(?P<NER>[A-Z-]+)\])$')

# more complex pattern that captures the lemma and token features
#token_re = re.compile(r'^(?:\\\*)?(?P<Token>.+?)(?:\*/)?(?:/(?P<Lemma>[^a-z]+))?(?:/(?P<POS>[A-Z]+)\d?)(?:#(?P<FEATS>[\w?-]+))?(?:\[(?P<NER>[A-Z-]+)\])$')

def preprocess_text(cintil_text):
    # delete unnecessary XML tags: <i>, </i>, <t>, </t>
    # some of these tags are opened without being properly closed
    cintil_text = re.sub(r'</?(i|t)> ', '', cintil_text)

    # URL encode '&' symbol - necessary in order to use a XML parser
    cintil_text = cintil_text.replace('&', '&amp;')

    # URL encode '<' and '>' symbols whenever they occur as tokens (not as XML tags)
    cintil_text = re.sub(r'(?:\\\*)?(<|>)(?:\*/)?/[^>]+?\[.+?\]', lambda match: escape(match.group(0)), cintil_text)

    return cintil_text

def create_rule(uncontracted, contracted):
    """
    Creates a rule composed of multiple uncontracted tokens as input
    and a single contracted token as output (POS tags are concatenated).
    """
    rule_in = [{'Token': token} for token in uncontracted.split()]
    rule_out = [{'Token': contracted, 'POS': {'CONCAT': '+'}}]
    rule = {'rule_in': rule_in, 'rule_out': rule_out}
    
    return rule

def clitic_rules(clitics):
    """Creates rules for clitic contractions in Portuguese."""
    rules = {}
    for name, contraction in clitics.items():
        uncontracted, contracted = contraction['uncontracted'], contraction['contracted']
        rules[name] = create_rule(uncontracted, contracted)

    return rules

def contraction_rules(contractions):
    """Creates rules for contractions in Portuguese."""
    rules = {}
    for name, contraction in contractions.items():
        uncontracted, contracted = contraction['uncontracted'], contraction['contracted']
        # add three rules for each contraction, a lowercase rule (e.g. de_ o -> do),
        # a capitalized rule (e.g. De_ o -> Do) and finally an uppercase rule 
        # (e.g. DE_ O -> DO), since all these three forms occur frequently in the data
        rules[name] = create_rule(uncontracted, contracted)
        rules[name.capitalize()] = create_rule(uncontracted.capitalize(),
                                                    contracted.capitalize())
        rules[name.upper()] = create_rule(uncontracted.upper(), contracted.upper())
                                                    
    return rules

# read CINTIL's file contents
dataset_in_path = dataset_in_dir / 'CINTIL-WRITTEN.txt'
cintil_text = dataset_in_path.read_text()

# preprocess text (required for the XML parser to work)
cintil_text = preprocess_text(cintil_text)

# parse XML document
cintil_doc = etree.fromstring(cintil_text)

# find the text of XML tags that correspond to sentences
sents = cintil_doc.xpath('excerpt/text/p/s/text()')

# read the sentences into a DataFrame
data_df = read_data(sents, token_re)

# convert rules to a format the RuleMatcher understands
contractions = contraction_rules(contractions)
clitics = clitic_rules(clitics)

# apply contraction, clitic and other rules in sequence
matcher = SequentialMatcher(fix_contractions, contractions, fix_clitics, clitics, rules)
data_df, rule_counts = matcher.apply_rules(data_df)

# convert POS tags to UD tagset
data_df['UPOS'] = data_df.POS.map(pos_tagset_ud_map)



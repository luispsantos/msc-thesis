import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from dataclasses import dataclass, field, InitVar
from typing import List, Set, Dict
from urlextract import URLExtract

class InvalidRuleFormat(Exception):
    '''Raise when rules do not adhere to the expected format.'''

class UnknownRuleOperation(Exception):
    '''Raise when rules attempt to compute unknown operations.'''

def _check_list_like(obj, error_msg):
    if not isinstance(obj, (list, tuple)):
        raise InvalidRuleFormat(f'{error_msg}: {obj} is not list-like.')

def _check_dict_like(obj, error_msg):
    if not isinstance(obj, dict):
        raise InvalidRuleFormat(f'{error_msg}: {obj} is not dict-like.')

def _check_dict_has_keys(obj, keys, error_msg):
    for key in keys:
        if key not in obj:
            raise InvalidRuleFormat(f'{error_msg}: {obj} is missing mandatory key {key}.')

@dataclass(order=True, frozen=True)
class Column:
    '''Stores an attribute-value pair for a specific column (e.g., Token.TEXT=="para").'''
    column: str
    attr: str
    value: object

    def __repr__(self):
        return f'{self.column}({self.attr}={self.value})'

@dataclass(order=True, frozen=True)
class Token:
    '''Stores a set of columns joined together by ANDs (e.g., Token=="os" and POS=="DET").'''
    columns: Set[Column] = field(init=False)
    token: InitVar[Dict]
    _val_types = {
        list: lambda val: tuple(val),
        dict: lambda val: tuple(val.items()),
        tuple: lambda val: val,
        str: lambda val: val,
        int: lambda val: val
    }

    def _parse_val(self, val):
        try:
            return self._val_types[type(val)](val)
        except KeyError:
            raise InvalidRuleFormat(f'Unknown type {type(val)} of {val}.')

    def _parse_token(self, token):
        columns = []
        for column, value in sorted(token.items()):
            if isinstance(value, str):
                columns.append(Column(column, 'TEXT', value))
            elif isinstance(value, dict):
                columns.extend(Column(column, attr, self._parse_val(val)) for attr, val in sorted(value.items()))
            else:
                raise InvalidRuleFormat(f'{value} is not str-like or dict-like.')

        return columns

    def __post_init__(self, token):
        _check_dict_like(token, 'Token should be a dict of columns and their values')
        object.__setattr__(self, 'columns', tuple(self._parse_token(token)))

    def __iter__(self):
        return iter(self.columns)

    def __repr__(self):
        return '+'.join(str(col) for col in self.columns)

@dataclass
class TokenList:
    '''Stores a list of tokens (e.g., (Token=="de" and POS=="ADP") and (Token=="os" and POS=="DET")).'''
    tokens: List[Token] = field(init=False)
    token_list: InitVar[List]

    def __post_init__(self, token_list):
        _check_list_like(token_list, 'Tokens should be a list of ordered tokens')
        tokens = tuple(Token(token) for token in token_list)
        object.__setattr__(self, 'tokens', tokens)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __repr__(self):
        return ', '.join(str(token) for token in self.tokens)

@dataclass
class Rule:
    '''Stores a pair (rule_in, rule_out). Every found instance of rule_in will be replaced by rule_out.'''
    name: str
    rule_in: TokenList = field(init=False)
    rule_out: TokenList = field(init=False)
    rule_properties: InitVar[Dict]

    def __post_init__(self, rule_properties):
        _check_dict_like(rule_properties, 'Rule properties should be a dictionary')
        _check_dict_has_keys(rule_properties, ['rule_in', 'rule_out'],
                             'Rule properties should contain keys rule_in and rule_out')

        rule_in = TokenList(rule_properties['rule_in'])
        rule_out = TokenList(rule_properties['rule_out'])
        object.__setattr__(self, 'rule_in', rule_in)
        object.__setattr__(self, 'rule_out', rule_out)

class RuleInOperation:
    '''Defines the possible column operations to be applied on a rule_in column.'''
    url_extractor = URLExtract()

    def __init__(self):
        self.col_no_nan_cache = {}

    def get_col_no_nan(self, data_col):
        col_name = data_col.name
        if col_name in self.col_no_nan_cache:
            col_no_nan = self.col_no_nan_cache[col_name]
        else:
            # replace NaN values from column with empty string
            # string operations are much quicker with category dtype
            col_no_nan = data_col.fillna('').astype('category')
            self.col_no_nan_cache[col_name] = col_no_nan

        return col_no_nan

    def binary_op1(self, col_func, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        col_mask = col_func(col_no_nan)
        col_masks = np.vstack([col_mask if value else ~col_mask for value in values])
        return col_masks

    def binary_op2(self, col_func, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        col_masks = np.vstack([col_func(col_no_nan, value) for value in values])
        return col_masks

    def is_lower_op(self, data_col, values):
        return self.binary_op1(lambda col: col.str.islower(), data_col, values)

    def is_upper_op(self, data_col, values):
        return self.binary_op1(lambda col: col.str.isupper(), data_col, values)

    def is_title_op(self, data_col, values):
        return self.binary_op1(lambda col: col.str.istitle(), data_col, values)

    def is_sent_boundary_op(self, data_col, values):
        return self.binary_op1(lambda col: col == '', data_col, values)

    def url_like_op(self, data_col, values):
        return self.binary_op1(lambda col: col.apply(lambda token: self.url_extractor \
                               .has_urls(token)).astype(bool), data_col, values)

    def starts_with_op(self, data_col, values):
        return self.binary_op2(lambda col, value: col.str.startswith(value), data_col, values)

    def ends_with_op(self, data_col, values):
        return self.binary_op2(lambda col, value: col.str.endswith(value), data_col, values)

    def contains_op(self, data_col, values):
        return self.binary_op2(lambda col, value: col.str.contains(value), data_col, values)

    def text_op(self, data_col, values):
        # create a boolean matrix of size N x T, indicating whether term t occurs
        # at position n. An efficient option is to use terms as categorical variables.
        category_dtype = CategoricalDtype(categories=values)
        col_masks = pd.get_dummies(data_col.astype(category_dtype), dtype=bool)

        col_masks = col_masks.values.T  # pd.DataFrame to transposed np.ndarray
        col_masks = np.ascontiguousarray(col_masks)  # optimal memory layout
        return col_masks

    def not_op(self, data_col, values):
        return ~self.text_op(data_col, values)

    def lower_op(self, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        return self.text_op(col_no_nan.str.lower(), values)

    def next_op(self, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        return self.text_op(col_no_nan.shift(-1), values)

    def in_op(self, data_col, values):
        # compute unique terms for the IN operator
        unique_values = list(set(term for value in values for term in value))
        col_masks = self.text_op(data_col, unique_values)

        col_terms = [[unique_values.index(term) for term in value] for value in values]
        col_masks = np.vstack([np.any(col_masks[col_idxs, :], axis=0) for col_idxs in col_terms])
        return col_masks

    def not_in_op(self, data_col, values):
        return ~self.in_op(data_col, values)

    def lower_in_op(self, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        return self.in_op(col_no_nan.str.lower(), values)

    def next_in_op(self, data_col, values):
        col_no_nan = self.get_col_no_nan(data_col)
        return self.in_op(col_no_nan.shift(-1), values)

    def apply_op(self, rule_in_op, data_col, values):
        try:
            return getattr(self, rule_in_op.lower() + '_op')(data_col, values)
        except AttributeError:
            raise UnknownRuleOperation(f'Unknown operation {rule_in_op}.')

class RuleOutOperation:
    '''Defines the possible column operations to be applied on a rule_out column.'''

    def concat_op(self, rule_in_array, token_idx, col_idx, value):
        rows = rule_in_array[:, :, col_idx].tolist()
        rule_in_array[:, token_idx, col_idx] = [value.join(row) for row in rows]  # value acts as a separator

    def delete_chars_op(self, rule_in_array, token_idx, col_idx, value):
        rule_in_array[:, token_idx, col_idx] = np.char.translate(rule_in_array[:, token_idx, col_idx].astype(str),
                                                                 {ord(char): None for char in value})

    def replace_op(self, rule_in_array, token_idx, col_idx, value):
        for old_str, new_str in value:
            rule_in_array[:, token_idx, col_idx] = np.char.replace(rule_in_array[:, token_idx, col_idx].astype(str),
                                                                   old_str, new_str if isinstance(new_str, str) else
                                                                   rule_in_array[:, new_str-1, col_idx])

    def text_op(self, rule_in_array, token_idx, col_idx, value):
        rule_in_array[:, token_idx, col_idx] = value

    def position_op(self, rule_in_array, token_idx, col_idx, value):
        rule_in_array[:, token_idx, col_idx] = rule_in_array[:, value-1, col_idx]

    def apply_ops(self, rule_in_array, rule, df_columns):
        column_idxs = {column: idx for idx, column in enumerate(df_columns)}
        try:
            for token_idx, token in enumerate(rule.rule_out):
                for col in token:
                    column, rule_out_op, value = col.column, col.attr, col.value
                    getattr(self, rule_out_op.lower() + '_op')(rule_in_array, token_idx, column_idxs[column], value)

        # other operations may be implemented in the future
        except AttributeError:
            raise UnknownRuleOperation(f'Unknown operation {rule_out_op}.')

class RuleMatcher:
    '''My own implementation of a rule-based matcher, loosely inspired on spaCy's Matcher.'''

    def __init__(self, rules=None):
        self.rules = []
        self.tokens_unique = set()
        self.columns_unique = set()

        self.add_rules(rules)

    def add_rules(self, rules):
        if not rules:
            return

        _check_dict_like(rules, 'Rules should be a dictionary of signature rule_name: rule_properties')
        new_rules = list(Rule(rule_name, rule_properties) for rule_name, rule_properties in rules.items())
        new_tokens = set(token for rule in new_rules for token in rule.rule_in)
        new_columns = set(column for rule in new_rules for token in rule.rule_in for column in token)

        self.rules.extend(new_rules)
        self.tokens_unique.update(new_tokens)
        self.columns_unique.update(new_columns)

        self.tokens_mapping = {token: idx for idx, token in enumerate(sorted(self.tokens_unique))}
        self.columns_mapping = {column: idx for idx, column in enumerate(sorted(self.columns_unique))}

    def clear_rules(self):
        # clear rules
        self.rules.clear()
        # clear uniques
        self.tokens_unique.clear()
        self.columns_unique.clear()
        # clear mappings
        self.tokens_mapping.clear()
        self.columns_mapping.clear()

    def _create_column_masks(self, data_df):
        # create a dictionary containing all possible column values
        # e.g. {'column1': {'attr1': ['val1', 'val2', ...], ...}, ...}
        column_values = {}
        for col in self.columns_mapping:
            column_values.setdefault(col.column, {}) \
                         .setdefault(col.attr, []).append(col.value)

        # combine column masks from different operations into a single matrix
        rule_in_operation = RuleInOperation()
        column_masks = np.vstack([rule_in_operation.apply_op(rule_in_op, data_df[column], values)
                                 for column, attrs in column_values.items() for rule_in_op, values in attrs.items()])

        return column_masks

    def _create_token_masks(self, column_masks):
        token_cols = [[self.columns_mapping[col] for col in token] for token in self.tokens_mapping]
        token_masks = np.vstack([np.all(column_masks[col_idxs], axis=0) if len(col_idxs) != 1
                                else column_masks[col_idxs[0]] for col_idxs in token_cols])

        return token_masks

    def _create_rule_masks(self, token_masks):
        rule_tokens = [[self.tokens_mapping[token] for token in rule.rule_in] for rule in self.rules]
        rule_masks = np.vstack([np.all(self._shift_align(token_masks, token_idxs), axis=0) if len(token_idxs) != 1
                               else token_masks[token_idxs[0]] for token_idxs in rule_tokens])

        return rule_masks

    def _shift_align(self, token_masks, token_idxs):
        data_size = token_masks.shape[1]
        col_length = data_size - len(token_idxs) + 1
        shifted_tokens = np.zeros((len(token_idxs), data_size), dtype=bool)
        # align shifted tokens (shift depends on the token position inside the rule)
        for idx, token_idx in enumerate(token_idxs):
            shifted_tokens[idx, :col_length] = token_masks[token_idx, idx:col_length+idx]

        return shifted_tokens

    def _merge_matches(self, rule_start_idxs, rule_in_length):
        # organize rules in tuples of (start_idx, rule_in_len, rule_idx) and sort the rules matches
        sorted_rules = sorted(((start_idx, rule_in_length[rule_idx], rule_idx)
                              for rule_idx, start_idxs in enumerate(rule_start_idxs)
                              for start_idx in start_idxs), key=lambda tup: tup[0])

        merged_rules = [sorted_rules[0]]
        for higher in sorted_rules[1:]:
            lower = merged_rules[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] < lower[0] + lower[1]:
                # pick the rule with the longest match (or largest rule_idx on matches with same length)
                upper_bound = max(lower[1:], higher[1:])
                merged_rules[-1] = (lower[0], *upper_bound)  # replace by merged rule
            else:
                merged_rules.append(higher)

        merged_start_idxs = [[] for rule_idx in range(len(rule_start_idxs))]
        for merged_rule in merged_rules:
            merged_start_idxs[merged_rule[2]].append(merged_rule[0])

        return merged_rules, merged_start_idxs

    def _apply_rules(self, rule_masks, df_values, df_index, df_columns):
        num_rules, num_tokens, num_columns = len(self.rules), len(df_values), len(df_columns)

        # compute length (number of tokens) for each (rule_in, rule_out) pair
        rule_in_length = [len(rule.rule_in) for rule in self.rules]
        rule_out_length = [len(rule.rule_out) for rule in self.rules]

        # compute start indices of the rule matches and merge overlapping matches
        rule_start_idxs = [rule_masks[rule_idx, :].nonzero()[0] for rule_idx in range(num_rules)]
        sorted_rules, rule_start_idxs = self._merge_matches(rule_start_idxs, rule_in_length)

        rule_out_operation = RuleOutOperation()
        index_mask = np.ones(num_tokens, dtype=bool)
        matched_rules = []

        for rule_idx, rule in enumerate(self.rules):
            rule_in_len, rule_out_len = rule_in_length[rule_idx], rule_out_length[rule_idx]

            # compute all token indices inside a rule match
            matched_token_idxs = [rule_start_idx+token_idx for rule_start_idx in rule_start_idxs[rule_idx]
                                  for token_idx in range(rule_in_len)]

            # compute all token indices to be dropped on rule_out
            dropped_token_idxs = [rule_start_idx+token_idx for rule_start_idx in rule_start_idxs[rule_idx]
                                  for token_idx in range(rule_out_len, rule_in_len)]

            # remove dropped tokens from the index
            index_mask[dropped_token_idxs] = False

            # compute rule_in array of size (num_matches, len(rule_in), num_columns)
            rule_in_array = df_values[matched_token_idxs].reshape(-1, rule_in_len, num_columns)

            # apply rule_out operations and reshape rule_in to the size of rule_out
            rule_out_operation.apply_ops(rule_in_array, rule, df_columns)
            rule_out_array = rule_in_array[:, :rule_out_len, :]
            matched_rules.append(rule_out_array)

        df_slices = []
        rule_counts = np.zeros(num_rules, dtype=int)

        df_slices.append(df_values[0:sorted_rules[0][0]])  # include initial slice before first match
        sorted_rules.append((num_tokens, None, None))  # include final slice after last match

        for (rule_start_idx, rule_in_len, rule_idx), (next_rule_start_idx, _, _) in zip(sorted_rules, sorted_rules[1:]):
            # append slice of matched rule
            rule_count = rule_counts[rule_idx]
            rule_counts[rule_idx] += 1
            df_slices.append(matched_rules[rule_idx][rule_count])

            # append dataframe slice in-between rules
            df_slices.append(df_values[rule_start_idx+rule_in_len:next_rule_start_idx])

        df_merged = pd.DataFrame(np.concatenate(df_slices, axis=0), index=df_index[index_mask], columns=df_columns)
        rule_counts = {rule.name: rule_counts[rule_idx] for rule_idx, rule in enumerate(self.rules)}

        return df_merged, rule_counts

    def apply_rules(self, data_df):
        column_masks = self._create_column_masks(data_df)
        token_masks = self._create_token_masks(column_masks)
        rule_masks = self._create_rule_masks(token_masks)
        data_df, rule_counts = self._apply_rules(rule_masks, data_df.values, data_df.index, data_df.columns)

        return data_df, rule_counts

class SequentialMatcher:
    '''A Matcher that applies groups of rules in sequence (rules are applied in the order they were added).'''
    
    def __init__(self, *rule_groups):
        self.matchers = []
        self.add_rules(*rule_groups)
        
    def add_rules(self, *rule_groups):
        self.matchers.extend(RuleMatcher(rules) for rules in rule_groups)
    
    def clear_rules(self):
        self.matchers.clear()
        
    def apply_rules(self, data_df):
        all_counts = {}
        for matcher in self.matchers:
            data_df, rule_counts = matcher.apply_rules(data_df)
            all_counts.update(rule_counts)
            
        return data_df, all_counts


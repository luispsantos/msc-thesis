import pandas as pd
from pathlib import Path
import numpy as np

class RuleMatcher:

    def __init__(self, rule_list=None, cols=['Token', 'POS']):
        self.rules = {}
        self.rule_in_max = 0
        self.rule_masks = set()
        self.cols = cols
        self.add_rules(rule_list)

    def add_rules(self, rule_list):
        if not rule_list:
            return

        new_rules = dict(self.evaluate_rule(rule_in, rule_out) for rule_in, rule_out in rule_list)
        new_rules_in_max = max(len(rule_in) for rule_in in new_rules.keys())

        self.rules.update(new_rules)
        self.rule_in_max = max(self.rule_in_max, new_rules_in_max)

    def evaluate_rule(self, rule_in, rule_out):
        rule_in_parsed, rule_out_parsed = self.parse_rule(rule_in), self.parse_rule(rule_out)
        self.rule_masks.add(tuple(tuple(col is not None for col in token) for token in rule_in_parsed))

        return rule_in_parsed, rule_out_parsed

    def parse_rule(self, rule):
        return tuple(tuple(token.get(col, None) for col in self.cols) for token in rule)

    def create_guesses(self, rows_shifted):
        for rule_mask in self.rule_masks:
            yield tuple(tuple(col if col_mask else None for col, col_mask in zip(token, token_mask)) for token, token_mask in zip(rows_shifted, rule_mask))

    def update_rows(self, rows_shifted, rule_out):
        return tuple(tuple(col_out if col_out is not None else col for col, col_out in zip(token, token_out)) for token, token_out in zip(rows_shifted, rule_out))

    def apply_rules(self, df):
        df_tuples = list(df.itertuples(index=False, name=None))
        data = []
        i = 0
        while i < len(df_tuples):
            for rule_in_guess in self.create_guesses(df_tuples[i:i+self.rule_in_max]):
                if rule_in_guess in self.rules:
                    rule_out = self.rules[rule_in_guess]
                    data.extend(self.update_rows(df_tuples[i:i+len(rule_out)], rule_out))
                    i += len(rule_in_guess)
                    break
            else:
                data.append(df_tuples[i])
                i += 1

        df_out = pd.DataFrame.from_records(data, columns=self.cols)
        return df_out


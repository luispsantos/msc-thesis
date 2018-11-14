rule_list = [
    # de + definite article
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'o', 'POS': 'DET'}], [{'Token': 'do', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'a', 'POS': 'DET'}], [{'Token': 'da', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'os', 'POS': 'DET'}], [{'Token': 'dos', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'as', 'POS': 'DET'}], [{'Token': 'das', 'POS': 'ADP+DET'}]),
    # em + definite article
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'o', 'POS': 'DET'}], [{'Token': 'no', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'a', 'POS': 'DET'}], [{'Token': 'na', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'os', 'POS': 'DET'}], [{'Token': 'nos', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'as', 'POS': 'DET'}], [{'Token': 'nas', 'POS': 'ADP+DET'}]),
    # a + definite article
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'o', 'POS': 'DET'}], [{'Token': 'ao', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'a', 'POS': 'DET'}], [{'Token': 'à', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'os', 'POS': 'DET'}], [{'Token': 'aos', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'as', 'POS': 'DET'}], [{'Token': 'às', 'POS': 'ADP+DET'}]),
    # por + definite article
    ([{'Token': 'por', 'POS': 'ADP'}, {'Token': 'o', 'POS': 'DET'}], [{'Token': 'pelo', 'POS': 'ADP+DET'}]),
    ([{'Token': 'por', 'POS': 'ADP'}, {'Token': 'a', 'POS': 'DET'}], [{'Token': 'pela', 'POS': 'ADP+DET'}]),
    ([{'Token': 'por', 'POS': 'ADP'}, {'Token': 'os', 'POS': 'DET'}], [{'Token': 'pelos', 'POS': 'ADP+DET'}]),
    ([{'Token': 'por', 'POS': 'ADP'}, {'Token': 'as', 'POS': 'DET'}], [{'Token': 'pelas', 'POS': 'ADP+DET'}]),

    # de + indefinite article
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'um', 'POS': 'DET'}], [{'Token': 'dum', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'uma', 'POS': 'DET'}], [{'Token': 'duma', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'uns', 'POS': 'DET'}], [{'Token': 'duns', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'umas', 'POS': 'DET'}], [{'Token': 'dumas', 'POS': 'ADP+DET'}]),
    # em + indefinite article
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'um', 'POS': 'DET'}], [{'Token': 'num', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'uma', 'POS': 'DET'}], [{'Token': 'numa', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'uns', 'POS': 'DET'}], [{'Token': 'nuns', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'umas', 'POS': 'DET'}], [{'Token': 'numas', 'POS': 'ADP+DET'}]),

    # de + personal pronoun
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'ele', 'POS': 'PRON'}], [{'Token': 'dele', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'ela', 'POS': 'PRON'}], [{'Token': 'dela', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'eles', 'POS': 'PRON'}], [{'Token': 'deles', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'elas', 'POS': 'PRON'}], [{'Token': 'delas', 'POS': 'ADP+PRON'}]),
    # em + personal pronoun
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'ele', 'POS': 'PRON'}], [{'Token': 'nele', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'ela', 'POS': 'PRON'}], [{'Token': 'nela', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'eles', 'POS': 'PRON'}], [{'Token': 'neles', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'elas', 'POS': 'PRON'}], [{'Token': 'nelas', 'POS': 'ADP+PRON'}]),
    # com + personal pronoun
    ([{'Token': 'com', 'POS': 'ADP'}, {'Token': 'mim', 'POS': 'PRON'}], [{'Token': 'comigo', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'com', 'POS': 'ADP'}, {'Token': 'ti', 'POS': 'PRON'}], [{'Token': 'contigo', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'com', 'POS': 'ADP'}, {'Token': 'si', 'POS': 'PRON'}], [{'Token': 'consigo', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'com', 'POS': 'ADP'}, {'Token': 'nós', 'POS': 'PRON'}], [{'Token': 'conosco', 'POS': 'ADP+PRON'}]),
    ([{'Token': 'com', 'POS': 'ADP'}, {'Token': 'vós', 'POS': 'PRON'}], [{'Token': 'convosco', 'POS': 'ADP+PRON'}]),

    # de + demonstrative pronoun (este)
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'este', 'POS': 'DET'}], [{'Token': 'deste', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'esta', 'POS': 'DET'}], [{'Token': 'desta', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'estes', 'POS': 'DET'}], [{'Token': 'destes', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'estas', 'POS': 'DET'}], [{'Token': 'destas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'isto', 'POS': 'PRON'}], [{'Token': 'disto', 'POS': 'ADP+PRON'}]),
    # em + demonstrative pronoun (este)
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'este', 'POS': 'DET'}], [{'Token': 'neste', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'esta', 'POS': 'DET'}], [{'Token': 'nesta', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'estes', 'POS': 'DET'}], [{'Token': 'nestes', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'estas', 'POS': 'DET'}], [{'Token': 'nestas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'isto', 'POS': 'PRON'}], [{'Token': 'nisto', 'POS': 'ADP+PRON'}]),

    # de + demonstrative pronoun (esse)
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'esse', 'POS': 'DET'}], [{'Token': 'desse', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'essa', 'POS': 'DET'}], [{'Token': 'dessa', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'esses', 'POS': 'DET'}], [{'Token': 'desses', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'essas', 'POS': 'DET'}], [{'Token': 'dessas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'isso', 'POS': 'PRON'}], [{'Token': 'disso', 'POS': 'ADP+PRON'}]),
    # em + demonstrative pronoun (esse)
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'esse', 'POS': 'DET'}], [{'Token': 'nesse', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'essa', 'POS': 'DET'}], [{'Token': 'nessa', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'esses', 'POS': 'DET'}], [{'Token': 'nesses', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'essas', 'POS': 'DET'}], [{'Token': 'nessas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'isso', 'POS': 'PRON'}], [{'Token': 'nisso', 'POS': 'ADP+PRON'}]),

    # de + demonstrative pronoun (aquele)
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aquele', 'POS': 'DET'}], [{'Token': 'daquele', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aquela', 'POS': 'DET'}], [{'Token': 'daquela', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aqueles', 'POS': 'DET'}], [{'Token': 'daqueles', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aquelas', 'POS': 'DET'}], [{'Token': 'daquelas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aquilo', 'POS': 'PRON'}], [{'Token': 'daquilo', 'POS': 'ADP+PRON'}]),
    # em + demonstrative pronoun (aquele)
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'aquele', 'POS': 'DET'}], [{'Token': 'naquele', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'aquela', 'POS': 'DET'}], [{'Token': 'naquela', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'aqueles', 'POS': 'DET'}], [{'Token': 'naqueles', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'aquelas', 'POS': 'DET'}], [{'Token': 'naquelas', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'aquilo', 'POS': 'PRON'}], [{'Token': 'naquilo', 'POS': 'ADP+PRON'}]),

    # a + demonstrative pronoun (aquele)
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'aquele', 'POS': 'DET'}], [{'Token': 'àquele', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'aquela', 'POS': 'DET'}], [{'Token': 'àquela', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'aqueles', 'POS': 'DET'}], [{'Token': 'àqueles', 'POS': 'ADP+DET'}]),
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'aquelas', 'POS': 'DET'}], [{'Token': 'àquelas', 'POS': 'ADP+DET'}]),
    # aquele can be PRON or DET (like outro and possibly others)

    # de + indefinite pronoun (outro)
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'outro', 'POS': 'DET'}], [{'Token': 'doutro', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'outra', 'POS': 'DET'}], [{'Token': 'doutra', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'outros', 'POS': 'DET'}], [{'Token': 'doutros', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'outras', 'POS': 'DET'}], [{'Token': 'doutras', 'POS': 'ADP+DET'}]),

    # em + indefinite pronoun (outro)
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'outro', 'POS': 'DET'}], [{'Token': 'noutro', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'outra', 'POS': 'DET'}], [{'Token': 'noutra', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'outros', 'POS': 'DET'}], [{'Token': 'noutros', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'outras', 'POS': 'DET'}], [{'Token': 'noutras', 'POS': 'ADP+DET'}]),

    # de + indefinite pronoun (algum)
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'algum', 'POS': 'DET'}], [{'Token': 'dalgum', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'alguma', 'POS': 'DET'}], [{'Token': 'dalguma', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'alguns', 'POS': 'DET'}], [{'Token': 'dalguns', 'POS': 'ADP+DET'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'algumas', 'POS': 'DET'}], [{'Token': 'dalgumas', 'POS': 'ADP+DET'}]),

    # em + indefinite pronoun (algum)
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'algum', 'POS': 'DET'}], [{'Token': 'nalgum', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'alguma', 'POS': 'DET'}], [{'Token': 'nalguma', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'alguns', 'POS': 'DET'}], [{'Token': 'nalguns', 'POS': 'ADP+DET'}]),
    ([{'Token': 'em', 'POS': 'ADP'}, {'Token': 'algumas', 'POS': 'DET'}], [{'Token': 'nalgumas', 'POS': 'ADP+DET'}]),

    # de + adverbs
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aqui', 'POS': 'ADV'}], [{'Token': 'daqui', 'POS': 'ADP+ADV'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'aí', 'POS': 'ADV'}], [{'Token': 'daí', 'POS': 'ADP+ADV'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'ali', 'POS': 'ADV'}], [{'Token': 'dali', 'POS': 'ADP+ADV'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'onde', 'POS': 'ADV'}], [{'Token': 'donde', 'POS': 'ADP+ADV'}]),
    ([{'Token': 'de', 'POS': 'ADP'}, {'Token': 'antes', 'POS': 'ADV'}], [{'Token': 'dantes', 'POS': 'ADP+ADV'}]),
    # donde is MWE in Google UD and ADV in Bosque
    # dantes is the reverse (MWE in Bosque and ADV in Google UD)

    # a + adverbs
    ([{'Token': 'a', 'POS': 'ADP'}, {'Token': 'onde', 'POS': 'ADV'}], [{'Token': 'aonde', 'POS': 'ADP+ADV'}]),
    # aonde is ADV in Google UD and Bosque

]

rule_list_reversed = [(rule_out, rule_in) for rule_in, rule_out in rule_list]

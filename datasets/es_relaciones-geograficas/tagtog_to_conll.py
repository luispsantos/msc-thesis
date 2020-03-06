import io,json
import nltk

class2tag = {'e_1':'Person',
            'e_10':'Clothing',
            'e_11':'Health',
            'e_12':'Gender',
            'e_13':'Warfare',
            'e_14':'Agriculture',
            'e_15':'Domesticated_Animal',
            'e_16':'Plants',
            'e_17':'Food',
            'e_18':'Metals',
            'e_19':'Architectural_materials',
            'e_2':'Place_Names',
            'e_20':'Disasters',
            'e_21':'Economy',
            'e_22':'Lithics',
            'e_24':'Aquatic_Animals',
            'e_25':'Mammals',
            'e_26':'Reptiles',
            'e_27':'Insects',
            'e_28':'Birds',
            'e_29':'Amphibians',
            'e_3':'Institutions',
            'e_30':'Terrestrial_routes',
            'e_31':'Aquatic_routes',
            'e_32':'Title',
            'e_33':'Language',
            'e_34':'Geopolitical_units',
            'e_35':'Ethnicity',
            'e_36':'Social_Classes',
            'e_37':'Material_Culture',
            'e_38':'Kinship',
            'e_39':'Resources',
            'e_4':'Profession',
            'e_40':'Activities',
            'e_41':'Deities',
            'e_5':'Date',
            'e_6':'Locations',
            'e_7':'Meteorology',
            'e_8':'Geo_Features',
            'e_9':'Ritual'
            }


def chunks(text_pos,grammar = """ NP: {<B-PER>+<I-PER>*} """):
        pattern = grammar
        chunker_fast = nltk.RegexpParser(pattern)
        chunk_result = chunker_fast.parse(text_pos)
        count = 0
        chunks = []
        for n in chunk_result:
            if isinstance(n, nltk.tree.Tree) and n.label() == 'NP':
                chunks.append((count,count+len(n)))
                count += len(n)
            else:
                count += 1
                continue
        return chunks


def gen_str_conlls_v1(data):
    #word and ner only!
    lines = ['-DOCSTART- -X- O O\n\n']
    for block in data:
        for w,iob in block:
            lines.append('{}\t_\t_\t{}\n'.format(w.replace(' ',''),iob))
        lines.append('\n')
    return ''.join(lines)



path_main_json = '../data/parsed'

with io.open('{}/sents_with_rule.json'.format(path_main_json)) as f_in:
    main_sents = [json.loads(line) for line in f_in]

import re
sents_out = []
for sent_data in main_sents:
    tok_tags = [x.split('_NER_') for x in sent_data['conll'].split()]
    temp = []
    for w,t in tok_tags:
        t = re.findall('e_\d+',t)
        if 'e_1' in t:
            temp.append((w,'PER'))
        elif 'e_2' in t and 'e_3' not in t:
            temp.append((w,'LOC'))
        elif 'e_6' in t and 'e_3' not in t:
            temp.append((w,'LOC'))
        elif 'e_3' in t:
            temp.append((w,'ORG'))
        else:
            temp.append((w,'O'))

    temp2 = [list(x) for x in temp]
    inds = chunks(temp,grammar = """ NP: {<PER>+} """)
    if inds:
        for st,end in inds:
            for i in range(st,end):
                if i == st:
                    temp2[i][1] = 'B-PER'
                else:
                    temp2[i][1] = 'I-PER'

    inds = chunks(temp,grammar = """ NP: {<ORG>+} """)
    if inds:
        for st,end in inds:
            for i in range(st,end):
                if i == st:
                    temp2[i][1] = 'B-ORG'
                else:
                    temp2[i][1] = 'I-ORG'



    inds = chunks(temp,grammar = """ NP: {<LOC>+} """)
    if inds:
        for st,end in inds:
            for i in range(st,end):
                if i == st:
                    temp2[i][1] = 'B-LOC'
                else:
                    temp2[i][1] = 'I-LOC'



    sents_out.append(temp2)




path_output = '../data/conll'

with io.open('{}/sents_PER_LOC_ORG.conll'.format(path_output),'w') as f_out:
    f_out.write(gen_str_conlls_v1(sents_out))



######
##  Split in train and test
######

import random
random.seed(10)



random.shuffle(sents_out)
n80 = int(len(sents_out)*0.80)
TRAIN_SENTS = sents_out[:n80]
TEST_SENTS = sents_out[n80:]



sents_train = [' '.join([w for w,_ in sent]).lower().strip() for sent in TRAIN_SENTS]
sents_test = [' '.join([w for w,_ in sent]).lower().strip() for sent in TEST_SENTS]

NEW_TEST_SET = []
for sent in TEST_SENTS:
    temp = ' '.join([w for w,_ in sent]).lower().strip()
    if temp in sents_train:
        print("!!!",sent)
        continue
    NEW_TEST_SET.append(sent)


path_output = '../data/conll'


with io.open('{}/train_PER_LOC_ORG.conll'.format(path_output),'w') as f_out:
    f_out.write(gen_str_conlls_v1(TRAIN_SENTS))
with io.open('{}/test_PER_LOC_ORG.conll'.format(path_output),'w') as f_out:
    f_out.write(gen_str_conlls_v1(NEW_TEST_SET))

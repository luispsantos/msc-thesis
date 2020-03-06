import io,json,sys
from lxml import html
from collections import defaultdict
import spacy

def chunks(text_pos,grammar = """ NP: {<PNM>+(<PREP|PREP\+DA><PNM>+)?} """):
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


nlp = spacy.load('es_core_news_sm') # to be used onlye as a tokenizer


# write file with the original together with a a pre-conll format
# Files shound be localted at ../data/annotated/doc_x.html
#

path_html_docs = "../data/annotated"
path_output = "../data/parsed"

with io.open('{}/sents_with_rule.json'.format(),'w') as f_out:

    for doc_id in range(1,6):
        html_fname = '{}/doc{}.html'.format(path_output,doc_id)

        with io.open(html_fname) as f:
            html_str = ''.join(f.readlines())
        tree = html.fromstring(html_str)
        sentences = tree.xpath('//p')

        # sents = defaultdict(lambda :defaultdict(list))
        annotated_sents = {}
        for node in sentences:
            sent_id = str(node.xpath('./@id')[0])
            text = str(node.xpath('.//text()')[0])
            annotated_sents[sent_id] = {'txt':text,'offsets':[]}

        with io.open('../data/annotated/doc{}.json'.format(doc_id)) as f:
            doc_json = json.load(f)

        main_sent_d = {'class_id':'','offsets':[]}
        for d_ent in doc_json['entities']:
            class_id = d_ent['classId']
            sent_id = d_ent['part']
            offsets = d_ent['offsets']
            for d in offsets:
                annotated_sents[sent_id]['offsets'].append((class_id,class2tag[class_id],d['start'],d['text']))
            annotated_sents[sent_id]['offsets'] = sorted(annotated_sents[sent_id]['offsets'],key = lambda x:x[2])

        count = 0
        for sent_id in annotated_sents:
            # print(sent_id)
            sent_txt = annotated_sents[sent_id]['txt']
            offsets = annotated_sents[sent_id]['offsets']

            for class_id,_,start,ent in offsets:
                end = start + len(ent)
                if sent_txt[start:end] != ent:
                    print([sent_txt])
                    print("doc{}".format(doc_id),sent_id,start,end,[sent_txt[start:end]],[ent])
                    sys.exit()

            doc = nlp(sent_txt)
            tok_tags = [[w.text,'O'] for w in doc]
            for i,w in enumerate(doc):

                for class_id,_,start,ent in offsets:
                    end = start + len(ent)
                    if w.idx >= start and w.idx < end:
                        if sent_txt[start:end] != ent:
                            print("error!")
                            sys.exit()
                        if tok_tags[i][1] == 'O':
                            tok_tags[i][1] = class_id
                        else:
                            if class_id != tok_tags[i][1]:
                                tok_tags[i].append(class_id)


            new_tok_tags = []
            for x in tok_tags:
                w = x[0]
                tags = x[1:]
                if w.strip():
                    new_tok_tags.append('_NER_'.join([w,'_'.join(tags)]))
            new_tok_tags = ' '.join(new_tok_tags)
            new_sent = {'doc_id':doc_id,'sent_id':sent_id,'original':sent_txt,'conll':new_tok_tags}
            f_out.write(json.dumps(new_sent,ensure_ascii = False))
            f_out.write('\n')

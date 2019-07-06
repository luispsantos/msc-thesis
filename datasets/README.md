# Sequence labeling datasets

A collection of sequence labeling datasets in Portuguese and Spanish, containing Parts-of-Speech (POS) and Named Entity Recognition (NER) labels on the domains of historical texts and modern, mostly newswire, corpora.
As the datasets were created in various research projects, they differ in terms of the content gathering process, tokenization strategies, tagset formats, and annotation guidelines.
Thus, an extensive normalization process was required to ensure a consistent format for all datasets.
All datasets were standardized as follows:
- Converted data layout to the [CoNLL format][CoNLL-03].
- Converted POS annotations to the [Universal POS tags][UPOS] of the Universal Dependencies (UD) project, featuring the additional tags for contractions: *ADP+DET*, *ADP+PRON*, *ADP+ADV* and *VERB+PRON*.
- Converted NER annotations to the [BIO tagging scheme][BIO-scheme], keeping only person (PER), location (LOC) and organization (ORG) named entities.

This repository contains the Python scripts to load and standardize the datasets.
Running the scripts requires Python 3.6+.
Due to licensing restrictions, we cannot distribute the code together with the original datasets.
We provide already pre-processed datasets [here][datasets-release], though we only grant the ZIP archive password for research purposes upon request.
We also provide our trained models [here][models-release], which were all trained on these datasets following different transfer learning techniques.
Please consider citing the original papers describing the datasets if you find this useful.

## Portuguese datasets

### Bosque
The Bosque corpus is part of the larger Floresta Sintática treebank, and contains newswire text in European Portuguese, from CETEMPúblico, and Brazilian Portuguese, from CETENFolha.
Bosque was fully revised by linguists and conversion rules were applied to convert the original data into the UD format.
- **URL**: https://universaldependencies.org/treebanks/pt_bosque/
- **Citation**: [Universal Dependencies for Portuguese](https://www.aclweb.org/anthology/W17-6523)

### CINTIL
CINTIL is a corpus of European Portuguese developed at the University of Lisbon.
It contains texts from written sources (e.g., newswire and literature) and spoken sources (e.g., telephone transcriptions, public/private conversations), annotated for named entities and POS tags.
- **URL**: http://cintil.ul.pt
- **Citation**: [Open Resources and Tools for the Shallow Processing of Portuguese:
The TagShare Project](http://lrec-conf.org/proceedings/lrec2006/pdf/311_pdf.pdf)

### CIPM
Corpus Informatizado do Português Medieval (CIPM) is a historical Portuguese corpus developed at Universidade Nova de Lisboa, containing early Portuguese texts from the 12th to the 16th centuries.
CIPM includes literary texts (e.g., travel narratives and doctrinal prose), and non-literary texts primarily of legal nature (e.g., private notarial documents and royal documents).
- **URL**: https://cipm.fcsh.unl.pt
- **Citation**: [O CIPM – Corpus Informatizado do Português Medieval](http://www.worldcat.org/oclc/7351606456)

### Colonia
The Colonia corpus of historical Portuguese consists of texts written between 1500 to 1936.
The corpus contains a balanced variety of 48 European Portuguese and 52 Brazilian Portuguese texts.
Word lemmas and POS tags were generated with TreeTagger, a probabilistic POS tagger reported to achieve accuracy higher than 95\%.
- **URL**: http://corporavm.uni-koeln.de/colonia/
- **Citation**: [Colonia: Corpus of Historical Portuguese](http://corporavm.uni-koeln.de/colonia/colonia.pdf)

### GSD
The GSD corpus of Brazilian Portuguese corresponds to annotated samples from news and blogs, converted from the legacy Google Universal Dependency Treebank.
- **URL**: https://universaldependencies.org/treebanks/pt_gsd/
- **Citation**: [Universal Dependency Annotation for Multilingual Parsing](https://www.aclweb.org/anthology/P13-2017)

### Mac-Morpho
The Mac-Morpho corpus was developed by the NILC group at the University of São Paulo and contains newswire text annotated with POS tags in Brazilian Portuguese.
The sentences were extracted from the newspaper Folha de São Paulo and cover a wide range of topics (e.g., agriculture, politics or sports).
- **URL**: http://nilc.icmc.usp.br/macmorpho/
- **Citation**: [An Account of the Challenge of Tagging a Reference Corpus for Brazilian Portuguese](https://link.springer.com/content/pdf/10.1007%2F3-540-45011-4.pdf)

### Paramopama
Paramopama is a manually revised corpus derived from the Portuguese WikiNER corpus.
The authors revised incorrectly assigned tags in an effort to improve upon the silver-standard WikiNER corpus.
The authors also created a version of the HAREM corpus in the CoNLL format.
- **URL**: https://github.com/davidsbatista/NER-datasets
- **Citation**: [Paramopama: a Brazilian-Portuguese Corpus for
Named Entity Recognition](http://www.lbd.dcc.ufmg.br/colecoes/eniac/2015/033.pdf)

### Post Scriptum
Post Scriptum is a historical Portuguese corpus developed by the Centro de Linguística da Universidade de Lisboa (CLUL) research group.
The corpus consists of informal letters in Portuguese and Spanish, most of which are unpublished, written between the 16th and early 20th centuries by authors from different social backgrounds.
Due to the nature of the letters, the textual contents are comparable to a spoken corpus, featuring issues from the everyday lives of people from past centuries.
This corpus is available both for Portuguese and Spanish.
- **URL**: http://ps.clul.ul.pt
- **Citation**: [Post Scriptum: Digital Archive of Everyday Writing](http://ps.clul.ul.pt/files/Papers-Congressos-PDFs/PostScriptumOslo(3).pdf)

### Tycho Brahe
The Tycho Brahe corpus of historical Portuguese consists of texts written by Portuguese authors born between 1380 and 1881.
A subset of these texts contain annotations for POS tags, currently comprising 47 texts with a total of 2.0M tokens.
- **URL**: http://www.tycho.iel.unicamp.br/corpus/

### WikiNER
WikiNER is a silver-standard corpus in multiple languages created from the link structure of Wikipedia.
This corpus is available both for Portuguese and Spanish.
- **URL**: https://doi.org/10.6084/m9.figshare.5462500.v1
- **Citation**: [Learning multilingual named entity recognition from Wikipedia](https://www.sciencedirect.com/science/article/pii/S0004370212000276)

## Spanish datasets

### AnCora
AnCora is a multilingual annotated corpus that comprises half a million words in Spanish and in Catalan that were annotated at the morphological, syntactic and semantic levels, thus containing POS and NER annotations.
The Spanish corpus consists mainly of newswire texts from the *EFE* Spanish news agency (225K words) and from the Spanish *El periódico* newspaper (200K words), including a smaller portion from the *Lexesp* Spanish balanced corpus (75K words).
- **URL**: http://stel.ub.edu/semeval2010-coref/
- **Citation**: [AnCora: Multilevel Annotated Corpora for Catalan and Spanish](https://pdfs.semanticscholar.org/0b01/90ddb5cff9861c47da7389dedfdbaa0b8f13.pdf)

### CoNLL-02
The CoNLL-02 shared task introduced two corpora with named entities in Spanish and Dutch.
The Spanish corpus consists of newswire articles from May 2000 made available by the Spanish *EFE* news agency.
- **URL**: http://www.lsi.upc.es/~nlp/tools/nerc/nerc.html
- **Citation**: [Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition](https://www.aclweb.org/anthology/W02-2024)

### Relaciones Geográficas
The Relaciones Geográficas historical corpus consists of answers to a questionnaire distributed to the dominions of King Philip II of Spain in the Viceroyalty of New Spain in North America.
The documents describe information with regard to 16th century ethnic groups in Mesoamerica, by considering questions regarding politics, the natural environment, population history, settlement patterns, native history and customs, etc.
- **URL**: https://www.lancaster.ac.uk/digging-ecm/corpus/

[CoNLL-03]: https://www.aclweb.org/anthology/W03-0419.pdf
[UPOS]: https://universaldependencies.org/u/pos/
[BIO-scheme]: https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)
[datasets-release]: https://github.com/luispsantos/msc-thesis/releases/tag/datasets
[models-release]: https://github.com/luispsantos/msc-thesis/releases/tag/models

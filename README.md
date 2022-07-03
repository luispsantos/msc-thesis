Source code and datasets supporting MSc dissertation:  
https://fenix.tecnico.ulisboa.pt/cursos/meic-a/dissertacao/846778572212377

**Simultaneous Tagging of Named Entities and Parts-of-Speech for Portuguese and Spanish Texts**  
**Evaluating Multi-Task and Cross-Language Neural Approaches**  

**Abstract:** Named entity recognition and parts-of-speech tagging are fundamental tasks in the field of natural language processing, currently with many practical applications. The current state-of-the-art approaches are based on the supervised training of deep neural networks, achieving a very high accuracy. However, when processing historical text or languages other than English (e.g., for processing data in Spanish and in Portuguese, independently of domain or time period), the fact that few training resources exist limits the use of modern machine learning approaches. To address this limitation, we collected and standardized a wide variety of datasets containing text in Portuguese and Spanish, annotated according to parts-of-speech and/or named entities. We then evaluated a modern architecture for sequence labeling, considering transfer learning approaches based on multi-task learning (i.e., simultaneously addressing parts-of-speech tagging and named entity recognition) and cross-lingual learning (i.e., aligning word embeddings of the Portuguese and Spanish languages in a single vector space), in order to jointly exploit all the available data and the underlying similarities on these tasks/languages, specifically to improve generalization on the smaller historical datasets. Our cross- lingual model achieves 91.97% of overall accuracy and 84.60% of entity-level F1 score for Portuguese, and 93.91% of overall accuracy and 64.34% of entity-level F1 score for Spanish, when averaging over all datasets for these languages.

The repository contains:
- A compilation of different datasets used for model training/testing, which is discussed [here|https://github.com/luispsantos/msc-thesis/tree/master/datasets] and available [here|https://github.com/luispsantos/msc-thesis/releases/tag/datasets].
- Adaptations on a previous [BiLSTM-CRF neural network model], together with several [pre-trained models|https://github.com/luispsantos/msc-thesis/releases/tag/models].
- Adaptations on a previous method for bulding cross-lingual word embeddings.

"""
A bidirectional LSTM with optional CRF and character-based presentation for NLP sequence tagging used for multi-task learning.

Author: Nils Reimers
License: Apache-2.0
"""

from __future__ import print_function
from util import BIOF1Validation

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
import math
import numpy as np
import sys
import gc
import time
import os
import random
import logging

from .keraslayers.ChainCRF import ChainCRF
from .keraslayers.Pentanh import Pentanh

class BiLSTM:
    def __init__(self, params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.model = None
        self.modelSavePath = None
        self.resultsSavePath = None


        # Hyperparameters for the network
        defaultParams = {'dropout': (0.5,0.5), 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam',
                         'charEmbeddings': None, 'charEmbeddingsSize': 30, 'charFilterSize': 30, 'charFilterLength': 3, 'charLSTMSize': 25, 'maxCharLength': 25,
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 5, 'miniBatchSize': 32,
                         'featureNames': ['tokens', 'casing'], 'addFeatureDimensions': 10}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def setMappings(self, mappings, embeddings):
        self.embeddings = embeddings
        self.mappings = mappings

    def setDataset(self, datasets, data):
        self.datasets = datasets
        self.data = data

        # Create some helping variables
        self.epoch = 0
        self.learning_rate_updates = {'sgd': {1: 0.1, 3: 0.05, 5: 0.01}}
        self.datasetNames = list(self.datasets.keys())
        self.evaluateDatasetNames = []
        self.labelKeys = {}
        self.tasks = []
        self.idx2Labels = {}
        self.trainSentences = None
        self.trainSentenceLengthRanges = None
        self.trainMiniBatchRanges = None

        for datasetName in self.datasetNames:
            labelKeys = self.datasets[datasetName]['label']
            self.labelKeys[datasetName] = labelKeys

            for labelKey in labelKeys:
                if labelKey not in self.tasks:
                    self.tasks.append(labelKey)
                    self.idx2Labels[labelKey] = {v: k for k, v in self.mappings[labelKey].items()}
            
            if self.datasets[datasetName]['evaluate']:
                self.evaluateDatasetNames.append(datasetName)
            
            logging.info("--- %s ---" % datasetName)
            logging.info("%d train sentences" % len(self.data[datasetName]['trainMatrix']))
            logging.info("%d dev sentences" % len(self.data[datasetName]['devMatrix']))
            logging.info("%d test sentences" % len(self.data[datasetName]['testMatrix']))
            
        self.casing2Idx = self.mappings['casing']
        self.numTrainTokens = {datasetName: sum(len(trainSent['tokens']) for trainSent in
                               self.data[datasetName]['trainMatrix']) for datasetName in self.datasetNames}

        totalTrainTokens = sum(self.numTrainTokens.values())
        # include sentence weights inversely proportional to the number
        # of tokens on each dataset (weights are used during training)
        for datasetName in self.datasetNames:
            trainData = self.data[datasetName]['trainMatrix']
            # function to compute sentence weights per dataset
            sentWeight = 1.0 - self.numTrainTokens[datasetName] / totalTrainTokens \
                         if len(self.datasetNames) != 1 else 1.0
            for trainSent in trainData:
                trainSent['weight'] = sentWeight

        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            logging.info("Pad words to uniform length for characters embeddings")
            all_sentences = []
            for datasetName in self.datasetNames:
                dataset = self.data[datasetName]
                for data in [dataset['trainMatrix'], dataset['devMatrix'], dataset['testMatrix']]:
                    for sentence in data:
                        all_sentences.append(sentence)

            self.padCharacters(all_sentences)
            logging.info("Words padded to %d characters" % (self.maxCharLen))

        
    def buildModel(self):
        tokens_input = Input(shape=(None,), dtype='int32', name='words_input')
        tokens = Embedding(input_dim=self.embeddings.shape[0], output_dim=self.embeddings.shape[1], weights=[self.embeddings], trainable=False, name='word_embeddings')(tokens_input)

        inputNodes = [tokens_input]
        mergeInputLayers = [tokens]

        for featureName in self.params['featureNames']:
            if featureName == 'tokens' or featureName == 'characters':
                continue

            feature_input = Input(shape=(None,), dtype='int32', name=featureName+'_input')
            feature_embedding = Embedding(input_dim=len(self.mappings[featureName]), output_dim=self.params['addFeatureDimensions'], name=featureName+'_embeddings')(feature_input)

            inputNodes.append(feature_input)
            mergeInputLayers.append(feature_embedding)
        

        # :: Character Embeddings ::
        if self.params['charEmbeddings'] not in [None, "None", "none", False, "False", "false"]:
            charset = self.mappings['characters']
            charEmbeddingsSize = self.params['charEmbeddingsSize']
            maxCharLen = self.maxCharLen
            charEmbeddings = []
            for _ in charset:
                limit = math.sqrt(3.0 / charEmbeddingsSize)
                vector = np.random.uniform(-limit, limit, charEmbeddingsSize)
                charEmbeddings.append(vector)

            charEmbeddings[0] = np.zeros(charEmbeddingsSize)  # Zero padding
            charEmbeddings = np.asarray(charEmbeddings)

            chars_input = Input(shape=(None, maxCharLen), dtype='int32', name='char_input')
            mask_zero = (self.params['charEmbeddings'].lower()=='lstm') #Zero mask only works with LSTM
            chars = TimeDistributed(
                Embedding(input_dim=charEmbeddings.shape[0], output_dim=charEmbeddings.shape[1],
                          weights=[charEmbeddings],
                          trainable=True, mask_zero=mask_zero), name='char_emd')(chars_input)

            if self.params['charEmbeddings'].lower()=='lstm':  # Use LSTM for char embeddings from Lample et al., 2016
                charLSTMSize = self.params['charLSTMSize']
                chars = TimeDistributed(Bidirectional(LSTM(charLSTMSize, activation='pentanh', recurrent_activation='pentanh',
                                                           return_sequences=False)), name="char_lstm")(chars)
            else:  # Use CNNs for character embeddings from Ma and Hovy, 2016
                charFilterSize = self.params['charFilterSize']
                charFilterLength = self.params['charFilterLength']
                chars = TimeDistributed(Conv1D(charFilterSize, charFilterLength, padding='same'), name="char_cnn")(chars)
                chars = TimeDistributed(GlobalMaxPooling1D(), name="char_pooling")(chars)

            self.params['featureNames'].append('characters')
            mergeInputLayers.append(chars)
            inputNodes.append(chars_input)
            
        if len(mergeInputLayers) >= 2:
            merged_input = concatenate(mergeInputLayers)
        else:
            merged_input = mergeInputLayers[0]
        
        # Add LSTMs
        shared_layer = merged_input
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = Bidirectional(LSTM(size, activation='pentanh', recurrent_activation='pentanh', return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]), name='shared_varLSTM_'+str(cnt))(shared_layer)
            else:
                """ Naive dropout """
                shared_layer = Bidirectional(LSTM(size, activation='pentanh', recurrent_activation='pentanh', return_sequences=True), name='shared_LSTM_'+str(cnt))(shared_layer)
                if self.params['dropout'] > 0.0:
                    shared_layer = TimeDistributed(Dropout(self.params['dropout']), name='shared_dropout_'+str(self.params['dropout'])+"_"+str(cnt))(shared_layer)
            
            cnt += 1
            
        outputs, losses = [], []

        # add softmax or CRF output layer (one classifier per task)
        for task in self.tasks:
            output = shared_layer
            classifier = self.params['classifier']
            n_class_labels = len(self.mappings[task])

            if classifier == 'Softmax':
                output = TimeDistributed(Dense(n_class_labels, activation='softmax'), name=task+'_softmax')(output)
                lossFct = 'sparse_categorical_crossentropy'
            elif classifier == 'CRF':
                output = TimeDistributed(Dense(n_class_labels, activation=None),
                                         name=task+'_hidden_lin_layer')(output)
                crf = ChainCRF(name=task+'_crf')
                output = crf(output)
                lossFct = crf.sparse_loss
            else:
                assert(False) #Wrong classifier

            outputs.append(output)
            losses.append(lossFct)
                
        # :: Parameters for the optimizer ::
        optimizerParams = {}
        if 'clipnorm' in self.params and self.params['clipnorm'] != None and  self.params['clipnorm'] > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']
        
        if 'clipvalue' in self.params and self.params['clipvalue'] != None and  self.params['clipvalue'] > 0:
            optimizerParams['clipvalue'] = self.params['clipvalue']
        
        if self.params['optimizer'].lower() == 'adam':
            opt = Adam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = Nadam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'rmsprop': 
            opt = RMSprop(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**optimizerParams)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **optimizerParams)

        model = Model(inputs=inputNodes, outputs=outputs)
        model.compile(loss=losses, optimizer=opt)

        model.summary(line_length=125)
        self.model = model

    def trainModel(self):
        self.epoch += 1

        if self.params['optimizer'] in self.learning_rate_updates and self.epoch in self.learning_rate_updates[self.params['optimizer']]:
            logging.info("Update Learning Rate to %f" % (self.learning_rate_updates[self.params['optimizer']][self.epoch]))
            K.set_value(self.model.optimizer.lr, self.learning_rate_updates[self.params['optimizer']][self.epoch])

        # train model a mini-batch at a time (1 epoch sees 1 batch of every dataset)
        for nnInput, nnLabels, nnWeights in self.minibatch_iterate_dataset():
            self.model.train_on_batch(nnInput, nnLabels, nnWeights)

    def minibatch_iterate_dataset(self):
        """ Create based on sentence length mini-batches with approx. the same size. Sentences and
        mini-batch chunks are shuffled and used to the train the model """
        
        if self.trainSentenceLengthRanges == None:
            """ Create mini batch ranges """
            # concatenate train sentences from all datasets
            trainSentences = []
            for datasetName in self.datasetNames:
                trainData = self.data[datasetName]['trainMatrix']
                # add mask labels for the tasks this dataset lacks
                for task in self.tasks:
                    if task not in self.labelKeys[datasetName]:
                        for sent in trainData:
                            sent[task] = [ChainCRF.mask_value] * len(sent['tokens'])

                trainSentences.extend(trainData)

            self.trainSentenceLengthRanges = []
            self.trainMiniBatchRanges = []

            # sort train sentences by sentence length
            trainSentences.sort(key=lambda sent: len(sent['tokens']))

            trainRanges = []
            oldSentLength = len(trainSentences[0]['tokens'])
            idxStart = 0

            #Find start and end of ranges for sentences with same length
            for idx in range(len(trainSentences)):
                sentLength = len(trainSentences[idx]['tokens'])

                if sentLength != oldSentLength:
                    trainRanges.append((idxStart, idx))
                    idxStart = idx

                oldSentLength = sentLength

            #Add last sentence
            trainRanges.append((idxStart, len(trainSentences)))

            #Break up ranges into smaller mini batch sizes
            miniBatchRanges = []
            for batchRange in trainRanges:
                rangeLen = batchRange[1]-batchRange[0]

                bins = int(math.ceil(rangeLen/float(self.params['miniBatchSize'])))
                binSize = int(math.ceil(rangeLen / float(bins)))

                for binNr in range(bins):
                    startIdx = binNr*binSize+batchRange[0]
                    endIdx = min(batchRange[1],(binNr+1)*binSize+batchRange[0])
                    miniBatchRanges.append((startIdx, endIdx))

            self.trainSentences = trainSentences
            self.trainSentenceLengthRanges = trainRanges
            self.trainMiniBatchRanges = miniBatchRanges

        #Shuffle sentences that have the same length
        trainSentences = self.trainSentences
        for start_idx, end_idx in self.trainSentenceLengthRanges:
            for i in reversed(range(start_idx+1, end_idx)):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = random.randint(start_idx, i)
                trainSentences[i], trainSentences[j] = trainSentences[j], trainSentences[i]

        #Shuffle the order of the mini batch ranges
        random.shuffle(self.trainMiniBatchRanges)

        #Iterate over the mini batch ranges
        for start_idx, end_idx in self.trainMiniBatchRanges:
            nnInput, nnLabels = [], []

            # features (inputs) of mini-batch
            for featureName in self.params['featureNames']:
                inputData = np.asarray([trainSentences[idx][featureName] for idx in range(start_idx, end_idx)])
                nnInput.append(inputData)

            # labels (outputs) of mini-batch
            for task in self.tasks:
                labels = np.asarray([trainSentences[idx][task] for idx in range(start_idx, end_idx)])
                labels = np.expand_dims(labels, -1)
                nnLabels.append(labels)

            # sentence (instance) weights of mini-batch
            nnWeights = np.asarray([trainSentences[idx]['weight'] for idx in range(start_idx, end_idx)])

            yield nnInput, nnLabels, nnWeights

    def storeResults(self, resultsFilepath):
        if resultsFilepath != None:
            directory = os.path.dirname(resultsFilepath)
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            self.resultsSavePath = open(resultsFilepath, 'w')
        else:
            self.resultsSavePath = None

    def fit(self, epochs):
        if self.model is None:
            self.buildModel()

        total_train_time = 0
        self.max_dev_score = {datasetName: {task: 0 for task in self.labelKeys[datasetName]}
                              for datasetName in self.evaluateDatasetNames}
        self.max_test_score = {datasetName: {task: 0 for task in self.labelKeys[datasetName]}
                              for datasetName in self.evaluateDatasetNames}

        no_improvement_since = 0
        max_early_stop_score = 0

        for epoch in range(epochs):      
            sys.stdout.flush()           
            logging.info("\n--------- Epoch %d -----------" % (epoch+1))

            start_time = time.time() 
            self.trainModel()
            time_diff = time.time() - start_time
            total_train_time += time_diff
            logging.info("%.2f sec for training (%.2f total)" % (time_diff, total_train_time))

            start_time = time.time()
            early_stop_score = 0

            for datasetName in self.evaluateDatasetNames:
                dev_matrix = self.data[datasetName]['devMatrix']
                test_matrix = self.data[datasetName]['testMatrix']

                dev_pred = self.predictLabels(dev_matrix)
                test_pred = self.predictLabels(test_matrix)

                dataset_tasks = self.labelKeys[datasetName]
                for task in dataset_tasks:
                    logging.info("-- %s - %s --" % (datasetName, task))

                    dev_true = [sentence[task] for sentence in dev_matrix]
                    test_true = [sentence[task] for sentence in test_matrix]

                    dev_score, test_score = self.computeScore(task, dev_pred[task], dev_true,
                                                              test_pred[task], test_true)

                    # add the dev score for each dataset - task pair
                    early_stop_score += len(dev_true) * dev_score

                    if dev_score > self.max_dev_score[datasetName][task]:
                        self.max_dev_score[datasetName][task] = dev_score
                        self.max_test_score[datasetName][task] = test_score

                    if self.resultsSavePath != None:
                        self.resultsSavePath.write("\t".join(map(str, [epoch + 1, datasetName, task, dev_score, test_score, self.max_dev_score[datasetName][task], self.max_test_score[datasetName][task]])))
                        self.resultsSavePath.write("\n")
                        self.resultsSavePath.flush()

                    logging.info("\nScores from epoch with best dev-scores:\n  Dev-Score: %.4f\n  Test-Score %.4f" % (self.max_dev_score[datasetName][task], self.max_test_score[datasetName][task]))
                    logging.info("")

            logging.info("%.2f sec for evaluation" % (time.time() - start_time))

            if early_stop_score > max_early_stop_score:
                no_improvement_since = 0
                max_early_stop_score = early_stop_score

                #Save the model
                if self.modelSavePath != None:
                    self.saveModel()
            else:
                no_improvement_since += 1

            if self.params['earlyStopping']  > 0 and no_improvement_since >= self.params['earlyStopping']:
                logging.info("!!! Early stopping, no improvement after "+str(no_improvement_since)+" epochs !!!")
                break

    def getSentenceLengths(self, sentences):
        sentenceLengths = {}
        for idx in range(len(sentences)):
            sentence = sentences[idx]['tokens']
            if len(sentence) not in sentenceLengths:
                sentenceLengths[len(sentence)] = []
            sentenceLengths[len(sentence)].append(idx)

        return sentenceLengths

    def predictLabels(self, sentences):
        predLabels = {task: [None]*len(sentences) for task in self.tasks}
        sentenceLengths = self.getSentenceLengths(sentences)

        for indices in sentenceLengths.values():
            nnInput = []                  
            for featureName in self.params['featureNames']:
                inputData = np.asarray([sentences[idx][featureName] for idx in indices])
                nnInput.append(inputData)

            taskPredictions = self.model.predict(nnInput, verbose=False)
            taskPredictions = [taskPredictions] if not isinstance(taskPredictions, list) else taskPredictions

            for taskIdx, task in enumerate(self.tasks):
                predictions = taskPredictions[taskIdx]
                predictions = predictions.argmax(axis=-1)  # obtain classes from one-hot encoding

                predIdx = 0
                for sentIdx in indices:
                    predLabels[task][sentIdx] = predictions[predIdx]
                    predIdx += 1

        return predLabels

    def computeScore(self, task, dev_pred, dev_true, test_pred, test_true):
        if task.endswith('_BIO') or task.endswith('_IOBES') or task.endswith('_IOB'):
            return self.computeF1Scores(task, dev_pred, dev_true, test_pred, test_true)
        else:
            return self.computeAccScores(task, dev_pred, dev_true, test_pred, test_true)

    def computeF1Scores(self, task, dev_pred, dev_true, test_pred, test_true):
        #train_pre, train_rec, train_f1 = self.computeF1(datasetName, self.datasets[datasetName]['trainMatrix'])
        #print "Train-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (train_pre, train_rec, train_f1)
        
        dev_pre, dev_rec, dev_f1 = self.computeF1(task, dev_pred, dev_true)
        logging.info("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (dev_pre, dev_rec, dev_f1))
        
        test_pre, test_rec, test_f1 = self.computeF1(task, test_pred, test_true)
        logging.info("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.4f" % (test_pre, test_rec, test_f1))
        
        return dev_f1, test_f1

    def computeAccScores(self, task, dev_pred, dev_true, test_pred, test_true):
        dev_acc = self.computeAcc(task, dev_pred, dev_true)
        test_acc = self.computeAcc(task, test_pred, test_true)

        logging.info("Dev-Data: Accuracy: %.4f" % (dev_acc))
        logging.info("Test-Data: Accuracy: %.4f" % (test_acc))

        return dev_acc, test_acc

    def computeF1(self, task, predLabels, correctLabels):
        idx2Label = self.idx2Labels[task]
        encodingScheme = task[task.index('_')+1:]

        pre, rec, f1 = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'O', encodingScheme)
        pre_b, rec_b, f1_b = BIOF1Validation.compute_f1(predLabels, correctLabels, idx2Label, 'B', encodingScheme)

        if f1_b > f1:
            logging.debug("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
            pre, rec, f1 = pre_b, rec_b, f1_b

        return pre, rec, f1

    def computeAcc(self, task, predLabels, correctLabels):
        numLabels = 0
        numCorrLabels = 0
        for sentenceId in range(len(correctLabels)):
            for tokenId in range(len(correctLabels[sentenceId])):
                numLabels += 1
                if correctLabels[sentenceId][tokenId] == predLabels[sentenceId][tokenId]:
                    numCorrLabels += 1
  
        return numCorrLabels/float(numLabels)

    def padCharacters(self, sentences):
        """ Pads the character representations of the words to the longest word in the dataset """
        #Find the longest word in the dataset
        maxCharLen = self.params['maxCharLength']
        if maxCharLen <= 0:
            for sentence in sentences:
                for token in sentence['characters']:
                    maxCharLen = max(maxCharLen, len(token))

        for sentenceIdx in range(len(sentences)):
            for tokenIdx in range(len(sentences[sentenceIdx]['characters'])):
                token = sentences[sentenceIdx]['characters'][tokenIdx]

                if len(token) < maxCharLen: #Token shorter than maxCharLen -> pad token
                    sentences[sentenceIdx]['characters'][tokenIdx] = np.pad(token, (0,maxCharLen-len(token)), 'constant')
                else: #Token longer than maxCharLen -> truncate token
                    sentences[sentenceIdx]['characters'][tokenIdx] = token[0:maxCharLen]

        self.maxCharLen = maxCharLen

    def saveModel(self):
        import json
        import h5py

        if self.modelSavePath == None:
            raise ValueError('modelSavePath not specified.')

        savePath = self.modelSavePath
        directory = os.path.dirname(savePath)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if os.path.isfile(savePath):
            logging.info(f'Model {savePath} already exists. Model will be overwritten')

        self.model.save(str(savePath), True)

        with h5py.File(savePath, 'a') as h5file:
            h5file.attrs['mappings'] = json.dumps(self.mappings)
            h5file.attrs['params'] = json.dumps(self.params)
            h5file.attrs['labelKeys'] = json.dumps(self.labelKeys)

    @staticmethod
    def loadModel(modelPath):
        import h5py
        import json
        from .keraslayers.ChainCRF import create_custom_objects

        model = keras.models.load_model(str(modelPath), custom_objects=create_custom_objects())

        with h5py.File(modelPath, 'r') as f:
            mappings = json.loads(f.attrs['mappings'])
            params = json.loads(f.attrs['params'])
            labelKeys = json.loads(f.attrs['labelKeys'])

        bilstm = BiLSTM(params)
        bilstm.setMappings(mappings, None)
        bilstm.model = model
        bilstm.labelKeys = labelKeys

        return bilstm

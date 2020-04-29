#!/bin/env python

import random
import numpy as np
import pandas as pd
from preprocess import TextPreprocess

# unused because of nfold validation
def train_test_split(x_dataset, y_dataset, split=0.8):
    # do not forget! columns are docs, so split columns in X_dataset
    break_at = int(split*len(y_dataset))
    x_train = x_dataset.iloc[:, :break_at]
    x_test = x_dataset.iloc[:, break_at:]
    y_train = y_dataset.iloc[:break_at]
    y_test = y_dataset.iloc[break_at:]
    return x_train, x_test, y_train, y_test


def wordFreqTable(x_train=None, y_train=None, processed_lines=None):
    # create a term frequency index
    word_freq = dict()
    word_freq['TOTAL'] = [0, 0] # as capital words wont exist in list, we can use TOTAL

    if processed_lines is not None:
        # if processed_lines is given in input
        for line in processed_lines:
            for word in line[0]:
                if word not in word_freq:
                    word_freq[word] = [0, 0] # no of words in class 0 and 1
                word_freq[word][line[1]] += 1
                word_freq['TOTAL'][line[1]] += 1
        return word_freq

    if x_train is not None and y_train is not None:
        # if dataframe is given as input
        for line in x_train.columns:
            for word in [ w for w in x_train.index if x_train.at[w, line] == 1 ]:
                if word not in word_freq:
                    word_freq[word] = [0, 0]
                word_freq[word][y_train.loc[line,0]] += 1
                word_freq['TOTAL'][y_train.loc[line,0]] += 1
        return word_freq


def predictClass(vector, word_freq, class_prob, alpha, out_of_vocab):
    if alpha == 0: alpha = 0.00001
    predictions = np.array([0]*len(class_prob))
    vector = vector[ vector == 1 ] # only select words which are present
    for clas in range(len(class_prob)):
        for word in vector.index:
            if word not in word_freq: continue # skip the words not in training dataset
            if not out_of_vocab and word_freq[word][clas] == 0: continue
            predictions[clas] += np.log((word_freq[word][clas] + alpha) / (word_freq['TOTAL'][clas] + alpha*len(word_freq)))
        predictions[clas] += np.log(class_prob[clas])
    return np.argmax(predictions) # returns the index/class which has highest prob


def runNaiveBayes(x_train, x_test, y_train, y_test, class_prob, smoothing, out_of_vocab):
    # bayes uses probability tables which are derived from training data
    word_freq_train = wordFreqTable(x_train=x_train, y_train=y_train)
    # predict class in testing dataset
    y_pred = [ predictClass(x_test[col], word_freq_train, class_prob, smoothing, out_of_vocab=out_of_vocab) for col in x_test.columns ]
    y_pred = np.array(y_pred)
    y_test = np.array(y_test[0])
    accuracy = (y_pred == y_test).sum() / len(y_test)
    TP = np.logical_and(y_pred == y_test, y_test == 1).sum()
    TN = np.logical_and(y_pred == y_test, y_test == 0).sum()
    FP = np.logical_and(y_pred != y_test, y_test == 0).sum()
    FN = np.logical_and(y_pred != y_test, y_test == 1).sum()
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f_score = (2*precision*recall)/(precision+recall)
    return accuracy, round(f_score, 4)


def nFoldNaiveBayes(n, processed_lines, smoothing, out_of_vocab):
    word_freq = wordFreqTable(processed_lines=processed_lines)
    # create a dataframe which hold vectors for each comment/line/response/feedback whatever you call it
    x_dataset = pd.DataFrame(0, index=word_freq.keys(), columns=[ i for i in range(len(processed_lines)) ])
    # build the dataframe vectors, aha the bag-of-words model from the days/code of IR!
    for i in range(len(processed_lines)):
        for word in processed_lines[i][0]:
            x_dataset.at[word, i] = 1
    # target variable for each line/comment
    y_dataset = pd.DataFrame([ i[1] for i in processed_lines ])
    # calculate class probabilities
    class_prob = [0, 0] # class probability, here, we have 2 classes only
    class_prob[1] = y_dataset.sum()[0]/len(y_dataset) # prob of class 1
    class_prob[0] = 1 - class_prob[1] # prob of class0 (complement)
    # train test split
    # x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, 0.8)
    slab_size = int(len(processed_lines)/n)
    start = 0
    end = slab_size
    accuracies = [0]*n
    for i in range(n):
        if i == n-1: end = len(processed_lines)
        x_train_slab = x_dataset.drop(x_dataset.columns[start:end], axis=1)
        x_test_slab = x_dataset.iloc[:,start:end]
        y_train_slab = y_dataset.drop(y_dataset.index[start:end], axis=0)
        y_test_slab = y_dataset.iloc[start:end]
        accuracies[i] = runNaiveBayes(x_train_slab, x_test_slab, y_train_slab, y_test_slab, class_prob, smoothing, out_of_vocab)
        print(f"Fold {i+1}: Accuracy: {accuracies[i][0]}, F-Score: {accuracies[i][1]}")
        start += slab_size
        end += slab_size
    f_scores = [ a[1] for a in accuracies ]
    accuracies = [ a[0] for a in accuracies ]
    meanAcc = round(np.mean(accuracies), 3)
    stdAcc = round(np.std(accuracies), 4)
    meanF = round(np.mean(f_scores), 3)
    stdF = round(np.std(f_scores), 4)
    return meanAcc, stdAcc, meanF, stdF


def main():

    # params
    nfold        = 5
    rm_punct     = True
    rm_stopwords = False
    stemming     = False
    lemmatize    = True
    smoothing    = 1
    out_of_vocab = True
    # split        = 0.8

    # read file/dataset
    with open('datasets/a1_d3.txt', 'r') as file:
        raw_lines = file.readlines()

    # preprocess lines and seperate target variable
    processed_lines = []
    for line in raw_lines:
        temp = TextPreprocess(line) # preprocess class constructor
        if rm_punct: temp.removePunctuation()   # comment to remove punctuation
        if rm_stopwords: temp.removeStopwords() # comment to keep stopwords
        if lemmatize: temp.lemmatize()
        if stemming: temp.stem()     # commment to NOT stem
        temp = temp.getTokens()
        processed_lines.append((temp[:len(temp)-1], int(temp[len(temp)-1])))
    # shuffle processed_lines if need arises (seed: 5)
    # random.shuffle(processed_lines)

    acc = nFoldNaiveBayes(nfold, processed_lines, smoothing=smoothing, out_of_vocab=out_of_vocab)
    print(f"Accuracy : {acc[0]*100}% ± {acc[1]*100}%")
    print(f"F-Score : {round(acc[2],3)} ± {round(acc[3],3)}")
    

if __name__ == "__main__":
    random.seed(42)
    main()

"""
5-Fold
RUN01:
    remove punct:       no
    remove stopwords:   no
    stemming:           no
    smoothing factor:   1
    train test split:   0.8
    ACCURACY:           73 + 6.6

RUN01:
    remove punct:       yes
    remove stopwords:   no
    stemming:           no
    smoothing factor:   0 / 1
    train test split:   0.8
    ACCURACY:           73 + 6.7 / 76 + 5.6

1-FOLD 
RUN01:
    remove punct:       no
    remove stopwords:   no
    stemming:           no
    smoothing factor:   1
    train test split:   0.7   / 0.8   / 0.9
    ACCURACY:           0.736 / 0.735 / 0.74

RUN02:
    remove punct:       yes
    remove stopwords:   no
    stemming:           no
    smoothing factor:   1
    train test split:   0.7  / 0.8   / 0.9
    ACCURACY:           0.79 / 0.785 / 0.79

RUN03:
    remove punct:       yes
    remove stopwords:   yes
    stemming:           no
    smoothing factor:   1
    train test split:   0.7  / 0.8  / 0.9
    ACCURACY:           0.76 / 0.78 / 0.72

RUN04:
    remove punct:       yes
    remove stopwords:   yes
    stemming:           yes
    smoothing factor:   1
    train test split:   0.7   / 0.8  / 0.9
    ACCURACY:           0.753 / 0.77 / 0.75

smoothing factor 1 gives better results (1~5% difference)
only punctuation removal gives best results 
"""
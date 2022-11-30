import json, re, math, random, time
from keras.preprocessing.text import Tokenizer
import numpy as np
from heapq import heappush, heappushpop
from scipy import spatial
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Lambda, dot, Input
random.seed(time.time())

#LOAD CORPUS
print('Loading corpus...')

corpus = []
maxLines = 500000
i = 0
for line in open('Electronics_5.json', 'r'):
    line = json.loads(line)
    line = re.sub(r"[\u0000-\u001f]", "", line['reviewText'].lower())
    corpus.append(line)
    i += 1
    if i == maxLines:
        break
corpus = [line for line in corpus if len(line.split()) > 1]

print('\tLoaded corpus.')

#TOKENIZE CORPUS
print('Tokenizing corpus...')

contentWindow = 2
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
vocabSize = len(tokenizer.word_index)+1
vocabDict = tokenizer.word_index
reverseDict = {}
for key in vocabDict:
    reverseDict[vocabDict[key]] = key
hiddenDimension = 100
corpus = tokenizer.texts_to_sequences(corpus)
corpus = tokenizer.sequences_to_texts(corpus)

for ind, line in enumerate(corpus):
    newLine = []
    for word in line.split():
        if word.isalpha():
            newLine.append(word)
    corpus[ind] = ' '.join(newLine)

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(corpus)
vocabSize = len(tokenizer.word_index)+1
vocabDict = tokenizer.word_index
reverseDict = {}
for key in vocabDict:
    reverseDict[vocabDict[key]] = key
corpus = tokenizer.texts_to_sequences(corpus)
corpus = tokenizer.sequences_to_texts(corpus)

for ind, line in enumerate(corpus):
    newLine = []
    for word in line.split():
        if tokenizer.word_counts[word] >= 5:
            newLine.append(word)
        else:
            newLine.append("<OOV>")
    corpus[ind] = ' '.join(newLine)

startPadding = ''
endPadding = ''

for i in range(contentWindow):
    startPadding += '<OOV> '
for i in range(contentWindow):
    endPadding += ' <OOV'

for index, line in enumerate(corpus):
    corpus[index] = startPadding + line + endPadding

corpus = tokenizer.texts_to_sequences(corpus)

print('\tTokenized corpus.')

#BUILD TRAINING SAMPLES
print('Creating training samples...')

frequencySum = 0
totalSum = 0

for token in tokenizer.word_counts.keys():
    totalSum += tokenizer.word_counts[token]
    frequencySum += math.pow(tokenizer.word_counts[token], 3/4)

def numOfTimes(num):
    return math.floor((math.pow(num, 3/4) / frequencySum) * totalSum)

unigramList = []
for token in tokenizer.word_counts.keys():
    for count in range(numOfTimes(tokenizer.word_counts[token])):
        unigramList.append(token)

contextWords = []
targetWords = []
labels = []

for line in corpus:
    size = len(line)
    for i in range(2, size-2):
        cw = []

        for j in range(i-2, i+contentWindow+1):
            if i != j:
                cw.append(line[j])

        tw = []
        tw.append(line[i])
        for k in range(0, 2*contentWindow-1):
            tw.append(0)
        targetWords.append(tw)
        contextWords.append(cw)
        labels.append(1)
targetWords = targetWords
contextWords = contextWords

negativeContextWords = []
negativeTargetWords = []
negativeLabels = []
for context in contextWords:
    ind = random.randint(0, len(unigramList)-1)
    negativeContextWords.append(context)
    tw = []
    tw.append(vocabDict[unigramList[ind]])
    for k in range(0, 2*contentWindow-1):
        tw.append(0)
    negativeTargetWords.append(tw)
    negativeLabels.append(0)

contextWords += negativeContextWords
targetWords += negativeTargetWords
labels += negativeLabels

print('\tCreated training samples.')

#CREATE MODEL
print('Running NN Model...')

contextInput = Input((2*contentWindow,))
targetInput = Input((2*contentWindow,))

embeddings = Embedding(vocabSize, hiddenDimension, input_length=2*contentWindow)
mean = Lambda(lambda x: K.sum(x, axis=1), output_shape=(hiddenDimension,))
contextEmbedding = embeddings(contextInput)
contextMean = mean(contextEmbedding)

targetEmbedding = embeddings(targetInput)
targetMean = mean(targetEmbedding)

dotProduct = dot([contextMean, targetMean], axes=1)
output = Dense(1, activation='sigmoid')(dotProduct)
cbowModel = Model([contextInput, targetInput], output)
cbowModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
cbowModel.fit(x=[np.array(contextWords), np.array(targetWords)], y=np.array(labels), batch_size=512, epochs=5)
finalEmbeddings = embeddings.get_weights()[0]

print('\tRan NN Model.')

#WRITE TO FILE
print('Writing to files...')

with open('cbowEmbeddings.txt', 'w') as file:
    for vector in finalEmbeddings:
        for feature in vector:
            file.write(str(feature))
            file.write(' ')
        file.write('\n')
file.close()

json.dump(vocabDict, open("cbowVocabDict.json", 'w'))
json.dump(reverseDict, open("cbowReverseDict.json", 'w'))

import json, re
from keras.preprocessing.text import Tokenizer
import scipy.sparse
from sklearn.decomposition import TruncatedSVD

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
vocabSize = len(tokenizer.word_index)
vocabDict = tokenizer.word_index
reverseDict = {}
for key in vocabDict:
    reverseDict[vocabDict[key]] = key
corpus = tokenizer.texts_to_sequences(corpus)
corpus = tokenizer.sequences_to_texts(corpus)

print('\tTokenized corpus.')

#BUILD CO-OCCURRENCE MATRIX
print('Building matrix...')

progressVariable = 0
coocurrenceMatrix = scipy.sparse.dok_matrix((vocabSize, vocabSize))

for line in corpus:
    line = line.split()
    for index, word in enumerate(line):

        if word not in vocabDict:
            continue

        pointer = index-1
        while (pointer >= max(0, index-contentWindow)):
            if line[pointer] in vocabDict:
                coocurrenceMatrix[vocabDict[word]-1, vocabDict[line[pointer]]-1] += 1
            else:
                print('ERROR: word not in vocabulary')
            pointer -= 1

        pointer = index+1
        while (pointer <= min(len(line)-1, index+contentWindow)):
            if line[pointer] in vocabDict:
                coocurrenceMatrix[vocabDict[word]-1, vocabDict[line[pointer]]-1] += 1
            else:
                print('ERROR: word not in vocabulary')
            pointer += 1

    progressVariable += 1
    if progressVariable % 50000 == 0:
        print('\tFound co-occurence for', progressVariable, 'vocabulary items so far.')

print('\tBuilt matrix.')

#APPLY SVD
print('Applying SVD...')

reducedDimension = 100

svd = TruncatedSVD(n_components=reducedDimension)
finalMatrix = svd.fit_transform(coocurrenceMatrix)

print('\tApplied SVD.')

#WRITE TO FILE
print('Writing to files...')

with open('svdEmbeddings.txt', 'w') as file:
    for vector in finalMatrix:
        for feature in vector:
            file.write(str(feature))
            file.write(' ')
        file.write('\n')

file.close()

json.dump(vocabDict, open("svdVocabDict.json", 'w'))
json.dump(reverseDict, open("svdReverseDict.json", 'w'))

print('\tWritten to files.')
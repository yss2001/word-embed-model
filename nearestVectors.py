import warnings
warnings.filterwarnings("ignore")

import sys
from scipy import spatial
from heapq import heappush, heappushpop
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

#CHECK COMMAND LINE ARGUMENTS
if len(sys.argv) < 2:
    print("No input file!")
    exit()

#LOAD EMBEDDINGS
print("Loading data...")
embeddings = []
with open(sys.argv[1], 'r') as file:
    for line in file:
        line = line.strip()
        embeddings.append(list(map(float, line.split())))
print("\tLoaded embeddings.")

#LOAD VOCABULARY
modelType = ''
if 'svd' in sys.argv[1]:
    modelType = 'svd'
else:
    modelType = 'cbow'
vocabDict = json.load(open(modelType + "VocabDict.json"))
reverseDict = json.load(open(modelType + "ReverseDict.json"))
print("\tLoaded dictionaries.")

#PRINT OUTPUT AND GRAPH
def visualize(word, labels, vectorList, vector):
    for l in labels:
        print(l)

    labels.append(word)
    vectorList.append(vector)
    
    tsne = TSNE(init='pca', learning_rate=200).fit_transform(vectorList)

    tx = tsne[:, 0]
    ty = tsne[:, 1]
    plt.scatter(tx, ty)
    plt.title('Nearest vectors of ' + str(word))
    for i, label in enumerate(labels):
        plt.annotate(label, (tx[i], ty[i]))
    plt.show()
    

#CALCULATE NEAREST VECTORS
def nearestVectors(word):
    top = []
    index = 0
    if modelType == 'svd':
        index = vocabDict[word]-1
    else:
        index = vocabDict[word]
    vector = embeddings[index]

    for ind, row in enumerate(embeddings):
        if modelType == 'cbow' and ind == 0:
            continue
        if ind != index:
            similarity = 1 - spatial.distance.cosine(vector, row)
            if similarity == 1 and all(v == 0 for v in row):
                print('Ignoring a vector with all 0 zeroes.')
                continue
            if len(top) < 10:
                heappush(top, (similarity, ind))
            else:
                heappushpop(top, (similarity, ind))

    top.sort(reverse=True)
    vectorList = []
    labels = []
    for t in top:
        if modelType == 'svd':
            labels.append(reverseDict[str(t[1]+1)])
        else:
            labels.append(reverseDict[str(t[1])])
        vectorList.append(embeddings[t[1]])

    visualize(word, labels, vectorList, vector)

#PRINT PROMPT
inp = ' '
print()
while True:
    inp = input('Enter word: ')
    if inp == 'Q':
        break
    if inp == 'R':
        ind = random.randint(0, len(vocabDict))
        inp = reverseDict[str(ind)]
        print('Random word from vocabulary:', inp)
    if inp not in vocabDict:
        print('Word not in vocabulary!')
    else:
        nearestVectors(inp)
    print()




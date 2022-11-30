## File locations
All the code files along with Electronics_5.json, cbowEmbeddings.json, cbowVocabDict.json, cbowReverseDict.json, svdEmbeddings.json, svdVocabDict.json, svdReverseDict.json must be in the same directory.

The embeddings are available in the link: https://drive.google.com/drive/folders/14SOye82pMk1Z7DTVbR4rUvbkUQ_MRGLj?usp=sharing

## Libraries
- gensim
- tensorflow
- scipy
- sklearn
- matplotlib

## Executing code
- Executing frequencyModel.py will compute the embeddings for the first model and outputs them to svdEmbeddings.txt. 

- Executing predictionModel.py will compute the embeddings for the second model and outputs them to cbowEmbeddings.txt

- Executing pretrained.py requires gensim and automatically downloads the dataset and prints the top 10 vectors.

- Executing nearestVectors.py requires passing in a command line argument which specifies the embedding filename. It will load the data and give a prompt where the user can enter a word, and outputs the top 10 nearest vectors along with 2D visualization. Pressing Q quits the prompt and R takes a random word from vocabulary and outputs.
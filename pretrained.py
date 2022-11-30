import gensim.downloader as api

info = api.info()
model = api.load("glove-wiki-gigaword-100")


print(model.most_similar("camera", topn=10))

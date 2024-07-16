# tokenazation - bir metindeki kelimeleri cümlelere veya cümleyi köklerine ayırır.
#sent_tokenize - cümleleri, word_tokenize ile kelimeleri

# TOKENAZATION
from nltk.tokenize import sent_tokenize,word_tokenize

text = "Alan Turing, İngiliz matematikçi, bilgisayar bilimcisi ve kriptolog. Bilgisayar biliminin kurucusu sayılır. Geliştirmiş oldugu Turing testi ile makinelerin ve bilgisayarların düşünme yetisine sahip olup olamayacakları konusunda bir kriter öne sürmüştür."

text.split()
word_tokenize(text)
sent_tokenize(text)

for token in word_tokenize(text):
    print(token)

# STOPWORDS
#corpus - doğal dil işlemede kullandığımız metine "corpus" denir.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = 'Fazıl Say is a Turkish pianist and composer who was born in Ankara, described recently as "not merely a pianist of genius; but undoubtedly he will be one of the great artists of the twenty-first century".'

sw = stopwords.words("english")

words = word_tokenize(text)

filtered_words = []
for word in words:
    if word not in sw:
        filtered_words.append(word)

filtered_words

# STEMMING - Bir kelimelinin kökünü almaya denir.Ama sadece kelimenin sonundaki ekleri kesiliyor.
from nltk.stem import PorterStemmer

ps = PorterStemmer()

words = ["drive","driving","driver","drives","drove","cats","children"]

for w in words:
    print(ps.stem(w))


# PART OF SPEECH TAGGING - Cümlenin Öğeleri

import nltk

text = 'Friedrich Wilhelm Nietzsche was a German philosopher, cultural critic, composer, poet, philologist, and a Latin and Greek scholar whose work has exerted a profound influence on Western philosophy and modern intellectual history. He began his career as a classical philologist before turning to philosophy. He became the youngest ever to hold the Chair of Classical Philology at the University of Basel in 1869 at the age of 24. Nietzsche resigned in 1879 due to health problems that plagued him most of his life; he completed much of his core writing in the following decade. In 1889 at age 44, he suffered a collapse and afterward, a complete loss of his mental faculties. He lived his remaining years in the care of his mother until her death in 1897 and then with his sister Elisabeth Förster-Nietzsche. Nietzsche died in 1900.'

tokenized = nltk.word_tokenize(text)

nltk.pos_tag(tokenized)

# NAMED ENTITY RECOGNITION -bir cümle içindeki kişi,organizasyon ve yer isimleri ve tarihler bilgisini bulur
import nltk

text = "Steve Jobs was an American entrepreneur and business magnate. He was the chairman, chief executive officer (CEO), and a co-founder of Apple Inc., chairman and majority shareholder of Pixar, a member of The Walt Disney Company's board of directors following its acquisition of Pixar, and the founder, chairman, and CEO of NeXT. Jobs is widely recognized as a pioneer of the microcomputer revolution of the 1970s and 1980s, along with Apple co-founder Steve Wozniak. "
tokenized = nltk.word_tokenize(text)
tagged = nltk.pos_tag(tokenized)
named_ent = nltk.ne_chunk(tagged)
named_ent.draw()

# LEMMATIZING - Kelimeleri köklerine ayırır.Daha karmaşık işlem uygulanır.kelimelerin morfolojik köklerine inilir.

from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

words = ["drive","driving","driver","drives","drove","cats","children"]

for w in words:
    print(lem.lemmatize(w))

lem.lemmatize("driving","v")

#%% word2vec

import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

f = open("hurriyet.txt", "r", encoding="utf8")
text = f.read()
t_list = text.split("\n")

corpus = []

for cumle in t_list:
    corpus.append(cumle.split())

print(corpus[:10])

model = Word2Vec(corpus, vector_size=100, window=5, min_count=5,sg=1)

model.wv["ankara"]

model.wv.most_similar("hollanda")

model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")


def closestwords_tsneplot(model, word):
    word_vectors = np.empty((0,100))
    word_labels = [word]

    close_words = model.wv.most_similar(word)
    
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)
    
    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors,np.array([model.wv[w]]), axis=0)
        
    tsne = TSNE(random_state=0, perplexity=10)
    Y = tsne.fit_transform(word_vectors)
    
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    
    plt.scatter(x_coords, y_coords)
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x,y), xytext=(5,-2), textcoords="offset points")

    plt.show()


closestwords_tsneplot(model, "berlin")

#%% glove

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

glove_input = "glove.6B.100d.txt"
word2vec_output = "glove.6B.100d.word2vec"
glove2word2vec(glove_input, word2vec_output)

model = KeyedVectors.load_word2vec_format(word2vec_output, binary=False)

model["istanbul"]

model.most_similar("nietzsche")

model.most_similar(positive=["woman","king"], negative=["man"],topn =1)

model.most_similar(positive=["ankara","germany"], negative=["berlin"],topn =1)





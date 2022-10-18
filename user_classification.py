from keras.models import load_model
from pymorphy2 import MorphAnalyzer
from nltk import regexp_tokenize, download
from nltk.corpus import stopwords
import pickle

download('stopwords')
stops = stopwords.words("russian")

model = load_model('MLP.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def normalize(sent, pat=r"(?u)\b\w\w+\b", morph=MorphAnalyzer()):
    return [morph.parse(tok)[0].normal_form if tok not in stops else ''
            for tok in regexp_tokenize(sent, pat)]


def prepare_text(text):
    text = text.map(lambda x: " ".join(normalize(x)))
    text = tokenizer.texts_to_sequences(text)
    text = tokenizer.sequences_to_matrix(text, mode='binary')
    return text


def classify(text):
    return model.predict(prepare_text(text))


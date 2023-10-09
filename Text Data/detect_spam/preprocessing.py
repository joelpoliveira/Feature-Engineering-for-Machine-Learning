import nltk
import string
from tqdm import tqdm


def get_words(document: list[str]):
    """
    Given a document, parses it's tokens, removing stopwords and punctuation;
    """
    stopwords = set(map(str.lower, nltk.corpus.stopwords.words("english")))
    punctuation = set(string.punctuation)
    for word in nltk.tokenize.word_tokenize(document):
        word = word.lower()
        if (word not in stopwords) and \
        (word.isalpha()):
            yield word

def remove_punct(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))            

def process_documents(docs):
    all_words = {}
    sentences = []
    index = 0

    for doc in tqdm(docs):
        for sentence in nltk.tokenize.sent_tokenize(doc):
            current_sentence = []
            sentence=remove_punct(sentence)
            for word in get_words(sentence):
                if word not in all_words:
                    all_words|= {word:index}
                    index+=1
                current_sentence.append(word)
            sentences.append(current_sentence)
    return all_words, sentences


def get_indexed_documents(docs, vocab, UNKOWN_TOKEN="[UNK]"):
    indexed_docs = []
    for doc in tqdm(docs):
        current_doc = []
        
        for sentence in nltk.tokenize.sent_tokenize(doc):
            sentence=remove_punct(sentence)
            
            for word in get_words(sentence):
                
                if word in vocab:
                    current_doc.append(vocab[word])
                else:
                    current_doc.append(vocab[UNKOWN_TOKEN])
        indexed_docs.append(current_doc)
    return indexed_docs

def get_max_length(index_docs):
    return max(list(map(len, index_docs)))


def pad(doc, MAX_LEN):
    diff = MAX_LEN - len(doc)
    if diff < 0:
        return doc[:MAX_LEN]
    return doc + [0]*diff


def apply_padding(docs, MAX_LEN):
    return list(map(lambda doc: pad(doc, MAX_LEN), docs, ))
    
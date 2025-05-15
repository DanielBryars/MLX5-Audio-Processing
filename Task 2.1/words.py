#Randall Monroe's list from https://xkcd.com/simplewriter/ 
#Downloaded from https://xkcd.com/simplewriter/words.js 
#curl -L https://xkcd.com/simplewriter/words.js -o words.js --output-dir . --fail

import spacy
import os

import spacy.cli
if not spacy.util.is_package("en_core_web_sm"):
    print("Installing spacy model")
    spacy.cli.download("en_core_web_sm")

def _get_thing_explainer_vocab_raw():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    words_path = os.path.join(script_dir, 'words.js')

    with open(words_path, encoding='utf-8') as f:
        l = f.read().split('=', 1)[1].strip().rstrip(';').strip('"').split('|')
        l.sort()
        return l

from spacy.tokens import Doc

def _clean(nlp, words):

    lemmatizer = nlp.get_pipe("lemmatizer")
    lemmas = set()

    for word in words:
        doc = Doc(nlp.vocab, words=[word])
        token = doc[0]
        token.pos = nlp.vocab.strings["VERB"]

        doc = lemmatizer(doc)
        lemma = doc[0].lemma_

        lemmas.add(lemma)

    return sorted(lemmas)

def get_thing_explainer_vocab():

    words = _get_thing_explainer_vocab_raw()
    print("Spacy Model Loaded")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    words = _clean(nlp, words)
    return words


if __name__ == "__main__":

    words = _get_thing_explainer_vocab_raw()
    print('\n'.join(words[:10]))
    print (f"raw_words count:{len(words)}")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    print("")
    print("lemma_")
    words = _clean(nlp, words)
    print('\n'.join(words[:10]))
    print (f"Lemmatised count:{len(words)}")
    #print('\n'.join(words))

    with open('words.txt', 'w') as f:
        for word in words:
            f.write(f"{word}\n")

    print("File written")




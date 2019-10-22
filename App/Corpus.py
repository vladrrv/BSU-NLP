import numpy as np
import re
import nltk
from nltk.tokenize import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('tagsets')

tag_descr = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition or subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List item marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund or present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb'
}
POS_TAGS = list(tag_descr.keys())


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def preprocess_text(text):
    text = text.replace("''", '"')
    text = text.replace("’", "'")
    return text


class Corpus:
    reg = r'[A-Za-z]+([\'|\-][A-Za-z]+)*'

    def __init__(self, raw_text='', freq_dict={}):
        self.raw_text = raw_text
        self.freq_dict = freq_dict

    def add_text(self, text):
        try:
            text = preprocess_text(text)

            self.raw_text += '\n\n**\n\n' + text
            tokenizer = TreebankWordTokenizer()
            spans = list(tokenizer.span_tokenize(text))

            words_ids = [(text[s:t], s) for s, t in spans if re.fullmatch(self.reg, text[s:t]) is not None]

            words = [word_id[0] for word_id in words_ids]
            words.sort(reverse=True)
            for word in words:
                self.add_word(word)
        except Exception as e:
            print(e)

    def get_words(self):
        return list(self.freq_dict.keys())

    def find_word_context(self, word):
        starts = [m.start() for m in re.finditer('(^|[^\w\-\'])(' + word + ')([^\w\-\']|$)', self.raw_text)]
        i = starts[0]
        context = self.raw_text[max(0, i - 100):i + 100]
        return "..." + context + "..."

    def add_word(self, word):
        if word[0].isupper():
            lower_word = word.lower()
            lower_val = self.freq_dict.get(lower_word, [0, None])[0]
            if lower_val > 0:
                self.freq_dict[lower_word][0] += 1
                return
        val = self.freq_dict.get(word, [0, None])[0]
        if val > 0:
            self.freq_dict[word][0] += 1
        else:
            self.freq_dict[word] = [1, {self.make_tag(word)}]

    def add_tag(self, word, tag):
        self.freq_dict[word][1].add(tag)

    def remove_tag(self, word, tag):
        self.freq_dict[word][1].remove(tag)

    def replace_word(self, old, new):
        l = len(old)
        starts = [m.start() for m in re.finditer('(^|[^\w\-\'])(' + old + ')([^\w\-\']|$)', self.raw_text)]
        new_text = ''
        prev_end = 0
        for start in starts:
            new_text += self.raw_text[prev_end:start] + new
            prev_end = start + l
        new_text += self.raw_text[prev_end:]
        self.raw_text = new_text

        cnt = self.freq_dict.pop(old)
        new_words = new.split()
        for w in new_words:
            if re.fullmatch(self.reg, w) is not None:
                self.add_word(w)

    def make_tag(self, word):
        tag = ''
        try:
            _, tag = nltk.pos_tag([word])[0]
        except Exception as e:
            print(e)
        return tag

    def get_freq(self, word):
        return self.freq_dict.get(word, (0, []))[0]

    def get_tags(self, word):
        return self.freq_dict.get(word, (0, []))[1]

    def get_init_form(self, word, tag):
        lemma = word
        try:
            lemmatizer = WordNetLemmatizer()
            wtag = get_wordnet_pos(tag)
            if wtag != '':
                lemma = lemmatizer.lemmatize(word, wtag)
        except Exception as e:
            print(e)
        return lemma

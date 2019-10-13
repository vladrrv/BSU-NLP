import numpy as np
import re
import nltk
from nltk.tokenize import *
nltk.download('averaged_perceptron_tagger')


def preprocess_text(text):
    text = text.replace("''", '"')
    text = text.replace("’", "'")
    return text


class Corpus:
    reg = "[A-Za-z]+([\'|\-][A-Za-z]+)*"

    def __init__(self, raw_text='', freq_dict={}):
        self.raw_text = raw_text
        self.freq_dict = freq_dict

    def add_text(self, text):
        text = preprocess_text(text)

        self.raw_text += '\n\n**\n\n' + text
        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(text))

        words_ids = [(text[s:t], s) for s, t in spans if re.fullmatch(self.reg, text[s:t]) is not None]

        words = [word_id[0] for word_id in words_ids]
        words.sort(reverse=True)

        for word in words:
            self.add_word(word)

    def get_words(self):
        return list(self.freq_dict.keys())

    def find_word_context(self, word):
        starts = [m.start() for m in re.finditer('(^|[^\w\-\'])('+word+')([^\w\-\']|$)', self.raw_text)]
        i = starts[0]
        context = self.raw_text[max(0, i-100):i+100]
        return "..."+context+"..."

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
            self.freq_dict[word] = [1, [self.make_tag(word)]]

    def replace_word(self, old, new):
        l = len(old)
        starts = [m.start() for m in re.finditer('(^|[^\w\-\'])('+old+')([^\w\-\']|$)', self.raw_text)]
        new_text = ''
        prev_end = 0
        for start in starts:
            new_text += self.raw_text[prev_end:start] + new
            prev_end = start+l
        new_text += self.raw_text[prev_end:]
        self.raw_text = new_text

        cnt = self.freq_dict.pop(old)
        new_words = new.split()
        reg = "[A-Za-z]+([\'|\-][A-Za-z]+)*"
        for w in new_words:
            if re.fullmatch(reg, w) is not None:
                self.add_word(w)

    def make_tag(self, word):
        tag = ''
        try:
            #print(nltk.pos_tag([word]))
            _, tag = nltk.pos_tag([word])[0]
        except Exception as e:
            print(e)

        return tag

    def get_freq(self, word):
        return self.freq_dict.get(word, (0, []))[0]

    def get_tag(self, word):
        return self.freq_dict.get(word, (0, []))[1][0]

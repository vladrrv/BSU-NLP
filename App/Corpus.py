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


class Entry:
    def __init__(self, corpus, word, tags=()):
        self.word = word
        self.init_forms = {tag : corpus.get_init_form(word, tag) for tag in tags}

    def __hash__(self):
        return hash(self.word)

    def get_init_for_tag(self, tag):
        return self.init_forms.get(tag, None)

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.__str__()


class Corpus:
    reg = r'[A-Za-z]+([\'|\-][A-Za-z]+)*'
    reg_find = r'(^|[^\w\-\'])({})([^\w\-\']|$)'
    sep = '\n\n**\n\n'

    def __init__(self):
        self.raw_text = ''
        self.tokenized_text = []
        self.freq_tag_dict = {}
        self.tokenizer = TreebankWordTokenizer()

    def is_valid_word(self, word, tag):
        return tag in POS_TAGS and re.fullmatch(self.reg, word)

    def add_text(self, text):
        try:
            text = preprocess_text(text)
            prev_len = len(self.raw_text)+len(self.sep)
            self.raw_text += self.sep + text
            spans = list(self.tokenizer.span_tokenize(text))
            spans_starts = [s+prev_len for s, t in spans]
            tokens = [text[s:t] for s, t in spans]
            print('making pos tags')
            tokens_tags = nltk.pos_tag(tokens)
            spans_tokens_tags = [(spans_starts[i], *tokens_tags[i]) for i in range(len(spans_starts))]
            self.tokenized_text += spans_tokens_tags
            print('done')

            words_tags = [(word, tag) for word, tag in tokens_tags if self.is_valid_word(word, tag)]

            words_tags.sort(reverse=True, key=lambda wt: wt[0])
            for word, tag in words_tags:
                self.add_word(word, tag)
        except Exception as e:
            print(e)

    def get_words(self):
        words = list(self.freq_tag_dict.keys())
        words.sort(key=lambda w: w.lower())
        return words

    def find_index(self, word):
        for i, (s, w, t) in enumerate(self.tokenized_text):
            if w == word:
                return i

    def find_word_context(self, word):
        index = self.find_index(word)
        num_words = 20
        start_i = max(0, index-num_words)
        end_i = min(len(self.tokenized_text), index+num_words)
        s, _, _ = self.tokenized_text[start_i]
        e, w, _ = self.tokenized_text[end_i]
        context = self.raw_text[s:e+len(w)]
        return "..." + context + "..."

    def add_word(self, word, tag):
        if word[0].isupper():
            lower_word = word.lower()
            lower_val = self.freq_tag_dict.get(lower_word, [0, None])[0]
            if lower_val > 0:
                self.freq_tag_dict[lower_word][0] += 1
                if self.freq_tag_dict[lower_word][1].get(tag) is None:
                    init = self.get_init_form(lower_word, tag)
                    self.freq_tag_dict[lower_word][1][tag] = init
                return
        val = self.freq_tag_dict.get(word, [0, None])[0]
        if val > 0:
            self.freq_tag_dict[word][0] += 1
            if self.freq_tag_dict[word][1].get(tag) is None:
                init = self.get_init_form(word, tag)
                self.freq_tag_dict[word][1][tag] = init
        else:
            init = self.get_init_form(word, tag)
            self.freq_tag_dict[word] = [1, {tag: init}]

    def add_tag(self, word, tag):
        if self.freq_tag_dict[word][1].get(tag) is None:
            init = self.get_init_form(word, tag)
            self.freq_tag_dict[word][1][tag] = init

    def remove_tag(self, word, tag):
        del self.freq_tag_dict[word][1][tag]

    def replace_word(self, old, new):
        l = len(old)
        l_new = len(new)
        delta = l_new - l

        # Replace in raw text
        starts = [m.start() for m in re.finditer(self.reg_find.format(old), self.raw_text)]
        start = starts[0]
        new_text = self.raw_text[0:start] + new + self.raw_text[start + l:]
        self.raw_text = new_text

        # Pop from dict
        if self.freq_tag_dict[old][0] == 1:
            self.freq_tag_dict.pop(old)
        else:
            self.freq_tag_dict[old][0] -= 1

        # Replace in tokenized text
        index = self.find_index(old)
        new_tokenized_text = self.tokenized_text[:index]

        new_spans = list(self.tokenizer.span_tokenize(new))
        new_tokens = []
        for s, t in new_spans:
            new_tokens.append(new[s:t])
        new_tokens_tags = nltk.pos_tag(new_tokens)
        s_old, _, _ = self.tokenized_text[index]
        for i in range(len(new_spans)):
            word, tag = new_tokens_tags[i]
            new_tokenized_text.append((new_spans[i][0]+s_old, word, tag))
            if self.is_valid_word(word, tag):
                self.add_word(word, tag)

        for i in range(index+1, len(self.tokenized_text)):
            s, word, tag = self.tokenized_text[i]
            new_tokenized_text.append((s+delta, word, tag))

        self.tokenized_text = new_tokenized_text

    def collect_stats(self):
        tag_freq = {tag: 0 for tag in POS_TAGS}
        word_tag_freq = {}
        tag_tag_freq = {}
        prev_wt = None
        for s, w, t in self.tokenized_text:
            if t not in POS_TAGS or re.fullmatch(self.reg, w) is None:
                continue
            tag_freq[t] += 1
            wt = (w.lower(), t)
            if word_tag_freq.get(wt):
                word_tag_freq[wt] += 1
            else:
                word_tag_freq[wt] = 1
            if prev_wt:
                tag_pair = (prev_wt[1], t)
                if tag_tag_freq.get(tag_pair):
                    tag_tag_freq[tag_pair] += 1
                else:
                    tag_tag_freq[tag_pair] = 1
            prev_wt = wt

        return tag_freq, word_tag_freq, tag_tag_freq

    def get_annotated_text(self):
        annotated_text = []
        prev_e = None
        for s, w, t in self.tokenized_text:
            e = s+len(w)
            if prev_e:
                annotated_text.append(self.raw_text[prev_e:s])
            annotated_text.append(self.raw_text[s:e])
            if t in POS_TAGS and re.fullmatch(self.reg, w):
                annotated_text.append('_')
                annotated_text.append(t)
            prev_e = e
        return ''.join(annotated_text)

    def get_freq(self, word):
        return self.freq_tag_dict.get(word, (0, {}))[0]

    def get_tags(self, word):
        return self.freq_tag_dict.get(word, (0, {}))[1].keys()

    @staticmethod
    def make_tag(word):
        tag = ''
        try:
            _, tag = nltk.pos_tag([word])[0]
        except Exception as e:
            print(e)
        return tag

    def get_init_form(self, word, tag):
        if self.freq_tag_dict.get(word) is not None and self.freq_tag_dict[word][1].get(tag) is not None:
            return self.freq_tag_dict[word][1][tag]
        lemma = word
        try:
            lemmatizer = WordNetLemmatizer()
            wtag = get_wordnet_pos(tag)
            if wtag != '':
                lemma = lemmatizer.lemmatize(word, wtag)
        except Exception as e:
            print(e)
        return lemma

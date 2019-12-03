import numpy as np
import re
import pickle
import string
import nltk
from nltk.tokenize import *
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from PyQt5.QtCore import *

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

tag_colormap = {
    'CC': 'MistyRose',
    'CD': 'MediumPurple',
    'DT': 'LightPink',
    'EX': 'Moccasin',
    'FW': 'HotPink',
    'IN': 'Lavender',
    'JJ': 'Aqua',
    'JJR': 'Aqua',
    'JJS': 'Aqua',
    'LS': 'LemonChiffon',
    'MD': 'LightCoral',
    'NN': 'GreenYellow',
    'NNS': 'GreenYellow',
    'NNP': 'PaleGreen',
    'NNPS': 'PaleGreen',
    'PDT': 'MediumSlateBlue',
    'POS': 'DarkSeaGreen',
    'PRP': 'LightSeaGreen',
    'PRP$': 'LightSeaGreen',
    'RB': 'Thistle',
    'RBR': 'Thistle',
    'RBS': 'Thistle',
    'RP': 'PowderBlue',
    'SYM': 'LemonChiffon',
    'TO': 'BlanchedAlmond',
    'UH': 'Tan',
    'VB': 'LightSalmon',
    'VBD': 'LightSalmon',
    'VBG': 'LightSalmon',
    'VBN': 'LightSalmon',
    'VBP': 'LightSalmon',
    'VBZ': 'LightSalmon',
    'WDT': 'LightGray',
    'WP': 'Silver',
    'WP$': 'Silver',
    'WRB': 'Gainsboro',
    'OTHER': 'LemonChiffon'
}


def get_color(tag):
    return tag_colormap[tag]


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
    text = text.replace("``", '"')
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    text = text.replace("’", "'")
    text = text.replace("\r", "")
    return text


class Corpus(QObject):

    status_sig = pyqtSignal(str)

    reg = r'[A-Za-z]+([\'|\-][A-Za-z]+)*'
    reg_find = r'(^|[^\w\-\'])({})([^\w\-\']|$)'
    sep = '\n\n**\n\n'

    html_span = "<span style=\"white-space: pre;\">{}</span>"
    html_colored_span = "<span style=\"background-color:{};\">{}</span>"

    def __init__(self):
        super().__init__()
        self.raw_text = ''
        self.text_spans = {}
        self.tokenized_text = []
        self.freq_tag_dict = {}
        self.tokenizer = TreebankWordTokenizer()
        self.sent_tokenizer = PunktSentenceTokenizer()
        self.stats = ({}, {}, {})
        self.refresh_stats = False

        self.modified_words = set()

    def load_from_pickle(self, pickle_file):
        with open(pickle_file, 'rb') as handle:
            data = pickle.load(handle)
            self.raw_text, self.text_spans, self.tokenized_text, self.freq_tag_dict, self.stats = data

    def save_to_pickle(self, pickle_file):
        with open(pickle_file, 'wb') as handle:
            data = self.raw_text, self.text_spans, self.tokenized_text, self.freq_tag_dict, self.stats
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def is_valid_word(self, word, tag):
        return tag in POS_TAGS and re.fullmatch(self.reg, word)

    def add_text(self, text, text_name='New text'):

        self.status_sig.emit(f'Adding "{text_name}" ...')
        try:
            print('Preprocessing...')
            text = preprocess_text(text)
            prev_len = len(self.raw_text)+len(self.sep)
            self.raw_text += self.sep + text

            default_name = text_name + ' ({})'
            count = 0
            while text_name in self.text_spans:
                count += 1
                text_name = default_name.format(count)
            self.text_spans[text_name] = (prev_len, prev_len + len(text))

            print('Tokenizing...')
            sent_spans = list(self.sent_tokenizer.span_tokenize(text))
            sents = []
            spans_starts = []
            for sent_start, sent_end in sent_spans:
                sent = text[sent_start:sent_end]
                tokens_sent = list(self.tokenizer.tokenize(sent))
                sents.append(tokens_sent)
                spans_sent = list(self.tokenizer.span_tokenize(sent))
                for i in range(len(spans_sent)):
                    s, e = spans_sent[i]
                    spans_starts.append(s+prev_len+sent_start)

            print('Making POS tags...')
            tokens_tags = []
            for tokens in sents:
                tokens_tags += nltk.pos_tag(tokens)
            spans_tokens_tags = [(spans_starts[i], *tokens_tags[i]) for i in range(len(spans_starts))]
            self.tokenized_text += spans_tokens_tags

            print('Filling dictionary...')

            words_tags = [(word, tag) for word, tag in tokens_tags if self.is_valid_word(word, tag)]

            words_tags.sort(reverse=True, key=lambda wt: wt[0])
            for word, tag in words_tags:
                self.add_word(word, tag)

            print('Done!')

        except Exception as e:
            print(e)

    def get_words(self):
        words = list(self.freq_tag_dict.keys())
        words.sort(key=lambda w: w.lower())
        modified = self.modified_words
        self.modified_words = set()
        return words, modified

    def find_index(self, word, num):
        count = 0
        wl = word.lower()
        for i, (_, w, _) in enumerate(self.tokenized_text):
            if w.lower() == wl:
                if count == num:
                    return i
                count += 1
        return None

    def find_word_by_raw_index(self, text_name, index):
        text_start, text_end = self.text_spans[text_name]
        index += text_start
        for i, (start, word, tag) in enumerate(self.tokenized_text):
            end = start + len(word)
            if start <= index < end:
                return i, word, tag if tag in POS_TAGS else 'OTHER'
            elif start > index:
                break
        return None, None, None

    def get_word_context(self, word, num=0):
        index = self.find_index(word, num)
        word_start, word, tag = self.tokenized_text[index]
        word_end = word_start + len(word)
        num_words = 20
        start_i = max(0, index-num_words)
        end_i = min(len(self.tokenized_text), index+num_words)
        context_start, _, _ = self.tokenized_text[start_i]
        context_end, _, _ = self.tokenized_text[end_i]
        context = [
            "...",
            self.html_span.format(self.raw_text[context_start:word_start]),
            self.html_colored_span.format(get_color(tag), word),
            self.html_span.format(self.raw_text[word_end:context_end]),
            "..."
        ]
        return "".join(context)

    def add_word(self, word, tag):
        self.refresh_stats = True
        self.modified_words.add(word)
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

    def replace_tag(self, index, new_tag):
        old_start, old_word, old_tag = self.tokenized_text[index]
        self.tokenized_text[index] = (old_start, old_word, new_tag)
        self.remove_tag(old_word, old_tag)
        self.add_tag(old_word, new_tag)

    def replace_word(self, index, new_word):
        old_start, old_word, old_tag = self.tokenized_text[index]
        l_old = len(old_word)
        l_new = len(new_word)
        delta = l_new - l_old

        # Replace in raw text
        new_text = self.raw_text[:old_start] + new_word + self.raw_text[old_start + l_old:]
        self.raw_text = new_text
        for text_name, (text_start, text_end) in self.text_spans.items():
            if text_start <= old_start < text_end:
                self.text_spans[text_name] = (text_start, text_end + delta)
            if old_start < text_start:
                self.text_spans[text_name] = (text_start + delta, text_end + delta)

        # Pop from dict
        old = old_word
        if old_word not in self.freq_tag_dict:
            old = old_word.lower()
        if self.freq_tag_dict[old][0] == 1:
            self.freq_tag_dict.pop(old)
        else:
            self.freq_tag_dict[old][0] -= 1

        # Replace in tokenized text
        new_tokenized_text = self.tokenized_text[:index]

        new_spans = list(self.tokenizer.span_tokenize(new_word))
        new_tokens = []
        for s, t in new_spans:
            new_tokens.append(new_word[s:t])
        new_tokens_tags = nltk.pos_tag(new_tokens)
        for i in range(len(new_spans)):
            word, tag = new_tokens_tags[i]
            new_tokenized_text.append((new_spans[i][0]+old_start, word, tag))
            if self.is_valid_word(word, tag):
                self.add_word(word, tag)

        for i in range(index+1, len(self.tokenized_text)):
            s, word, tag = self.tokenized_text[i]
            new_tokenized_text.append((s+delta, word, tag))

        self.tokenized_text = new_tokenized_text

    def collect_stats(self):
        if self.refresh_stats:
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
            self.stats = tag_freq, word_tag_freq, tag_tag_freq
            self.refresh_stats = False
        return self.stats

    def get_annotated_text(self, text_name):
        annotated_text = []
        colored_text = []
        text_start, text_end = self.text_spans[text_name]
        prev_e = text_start
        for s, w, t in self.tokenized_text:
            if text_start > s:
                continue
            if s >= text_end:
                break
            w = "\"" if w == "``" or w == "''" else w
            e = s+len(w)
            trash = self.raw_text[prev_e:s]
            word = self.raw_text[s:e]
            if t in POS_TAGS and w not in string.punctuation:
                tag = t
            else:
                tag = 'OTHER'
            annotated_text.append((trash, word, tag))
            colored_text.append(self.html_span.format(trash))
            color = get_color(tag)
            colored_text.append(self.html_colored_span.format(color, word))
            prev_e = e

        return ''.join(colored_text)

    def get_text_names(self):
        return list(self.text_spans.keys())

    def get_raw_text(self, text_name):
        text_start, text_end = self.text_spans[text_name]
        return self.raw_text[text_start:text_end]

    def get_word_bounds(self, text_name, index):
        text_start, _ = self.text_spans[text_name]
        start, word, _ = self.tokenized_text[index]
        end = start + len(word)
        return start - text_start, end - text_start

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

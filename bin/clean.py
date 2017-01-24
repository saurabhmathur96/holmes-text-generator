from __future__ import print_function, unicode_literals
import nltk
import sys
import re 
import string
import codecs
from collections import Counter
from utils.config import *

UTF8Writer = codecs.getwriter("utf8")
sys.stdout = UTF8Writer(sys.stdout)

punctuation_regex = re.compile("\s*([{0}])\s*".format(re.escape(string.punctuation)))

all_text = open(RAW_FILE).read().decode("utf8")
sentences = nltk.sent_tokenize(all_text)

def clean_sentence(s):
    s = " ".join(nltk.word_tokenize(s))
    s = s.replace("``", "\"").replace("''", "\"")
    s = re.sub(punctuation_regex, r"\1", s)
    s = s.replace(",", ", ")
    s = s.replace("`", "").replace("*", "")
    s = s.replace(" n't", "n't")
    return s

cleaned = " ".join(clean_sentence(s).encode("ascii", "ignore") for s in sentences).lower()

open(DATA_FILE, "w").write(cleaned)

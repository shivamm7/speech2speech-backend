from mosestokenizer import MosesSentenceSplitter, MosesTokenizer, MosesDetokenizer

from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import codecs

from subword_nmt.apply_bpe import BPE

import ctranslate2

# En
## Split
splitsents_en = MosesSentenceSplitter('en')
## Tokenize
tokenize_en = MosesTokenizer('en')

# En-Hi
codes_en_hi = codecs.open("MT/en/hi/v1/bpe-codes/codes.en", encoding='utf-8')
bpe_en_hi = BPE(codes_en_hi)
translator_en_hi = ctranslate2.Translator("MT/en/hi/v1/model_deploy", inter_threads=4, intra_threads=1)

# En-Mr
codes_en_mr = codecs.open("MT/en/mr/v1/bpe-codes/codes.en", encoding='utf-8')
bpe_en_mr = BPE(codes_en_mr)
translator_en_mr = ctranslate2.Translator("MT/en/mr/v1/model_deploy", inter_threads=4, intra_threads=1)

# Hi-Mr
## Normalize
factory=IndicNormalizerFactory()
normalizer_hi=factory.get_normalizer("hi",remove_nuktas=False)
## BPE
codes_hi_mr = codecs.open("MT/hi/mr/v1/bpe-codes/codes.hi", encoding='utf-8')
bpe_hi_mr = BPE(codes_hi_mr)
## Translate
translator_hi_mr = ctranslate2.Translator("MT/hi/mr/v1/model_ct2", inter_threads=4, intra_threads=1)

# Mr-Hi
## Normalize
normalizer_mr=factory.get_normalizer("mr",remove_nuktas=False)
## BPE
codes_mr_hi = codecs.open("MT/mr/hi/v1/bpe-codes/codes.mr", encoding='utf-8')
bpe_mr_hi = BPE(codes_mr_hi)
## Translate
translator_mr_hi = ctranslate2.Translator("MT/mr/hi/v1/model_ct2", inter_threads=4, intra_threads=1)

def mt_en_hi(source_sentence):

    # Lowercase
    source_sentence = source_sentence.lower()

    # Tokenize
    source_sentence = ' '.join(tokenize_en(source_sentence))

    # Apply BPE
    source_sentence = bpe_en_hi.process_line(source_sentence).split(" ")

    # Translate
    target_sentence = translator_en_hi.translate_batch([source_sentence], beam_size=5, max_batch_size=16)

    # Remove BPE
    target_sentence = (' '.join(target_sentence[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Detokenize Output
    target_sentence = indic_detokenize.trivial_detokenize_indic(target_sentence)

    return target_sentence

def mt_en_mr(source_sentence):
	# Lowercase
    source_sentence = source_sentence.lower()

    # Tokenize
    source_sentence = ' '.join(tokenize_en(source_sentence))

    # Apply BPE
    source_sentence = bpe_en_mr.process_line(source_sentence).split(" ")

    # Translate
    target_sentence = translator_en_mr.translate_batch([source_sentence], beam_size=5, max_batch_size=16)

    # Remove BPE
    target_sentence = (' '.join(target_sentence[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Detokenize Output
    target_sentence = indic_detokenize.trivial_detokenize_indic(target_sentence)

    return target_sentence

def mt_hi_mr(source_sentence):

    # Normalize
    source_sentence = normalizer_hi.normalize(source_sentence)

    # Tokenize
    source_sentence = ' '.join(indic_tokenize.trivial_tokenize(source_sentence))

    # Apply BPE
    source_sentence = bpe_hi_mr.process_line(source_sentence).split(" ")

    # Translate
    target_sentence = translator_hi_mr.translate_batch([source_sentence], beam_size=5, max_batch_size=16)

    # Remove BPE
    target_sentence = (' '.join(target_sentence[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Detokenize Output
    target_sentence = indic_detokenize.trivial_detokenize_indic(target_sentence)

    return target_sentence


def mt_mr_hi(source_sentence):

    # Normalize
    source_sentence = normalizer_mr.normalize(source_sentence)

    # Tokenize
    source_sentence = ' '.join(indic_tokenize.trivial_tokenize(source_sentence))

    # Apply BPE
    source_sentence = bpe_mr_hi.process_line(source_sentence).split(" ")

    # Translate
    target_sentence = translator_mr_hi.translate_batch([source_sentence], beam_size=5, max_batch_size=16)

    # Remove BPE
    target_sentence = (' '.join(target_sentence[0].hypotheses[0]) + " ").replace("@@ ", "")

    # Detokenize Output
    target_sentence = indic_detokenize.trivial_detokenize_indic(target_sentence)

    return target_sentence
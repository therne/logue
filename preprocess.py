
from itertools import chain
from konlpy.tag import Twitter

FILTER_JOSA = False

# we'll use Twitter Korean tagger (https://openkoreantext.org/)
# because it's fast and appropriate to internet conversations.
korean_tagger = Twitter()


def preprocess(utters):
    tokenized_utters = [preprocess_sentence(sentence) for sentence in utters]
    max_sentence_len = max_length_of(tokenized_utters)
    max_token_len = max_length_of(flatten(tokenized_utters))

    return {
        'utters': tokenized_utters,
        'max_sentence_len': max_sentence_len,
        'max_token_len': max_token_len
    }


def preprocess_sentence(sentence):
    # korean needs to be tokenized and converted into stems
    # because of conjugation in korean.
    tagged_tokens = korean_tagger.pos(sentence, stem=True)

    if FILTER_JOSA:
        tagged_tokens = filter(lambda t: t[1] != 'Josa', tagged_tokens)

    return [morph for morph, tag in tagged_tokens]


max_length_of = lambda x: max([len(elem) for elem in x])
flatten = chain.from_iterable

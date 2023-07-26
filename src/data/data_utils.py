import re
from difflib import SequenceMatcher

import codecs
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from text_unidecode import unidecode
from tqdm.notebook import tqdm
import language_tool_python
from readability import Readability


def add_grammar_mistake(text):
    tool = language_tool_python.LanguageTool('en-US')
    res = text.map(lambda x: len(tool.check(x)))
    tool.close()
    mini, maxi = min(res), max(res)
    nmin, nmax = 1, 5
    return res.map(lambda x: ((x - mini) / maxi) * (nmax - nmin) + nmin)


def add_readability(text):
    res = text.map(lambda x: Readability(x).flesch().score)
    mini, maxi = min(res), max(res)
    nmin, nmax = 1, 5
    return res.map(lambda x: ((x - mini) / maxi) * (nmax - nmin) + nmin)


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


# Register the encoding and decoding error handlers for `utf-8` and `cp1252`.
codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)



def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def replace_newline(text):
    text = text.replace('\r\n', '[BR] ')
    text = text.replace('\\u', '[SU] ')
    return text


def preprocess_text(text):
    text = text.replace(u'\x9d', u' ')
    text = resolve_encodings_and_normalize(text)
    text = replace_newline(text)
    return text.strip()



def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def get_span_from_text(text, span='sentence'):
    if span == 'sentence_piece':
        return get_sentence_piece_span(text)
    elif span == 'sentence':
        return get_sentence_span(text)
    else:
        return get_paragraph_span(text)


def get_sentence_piece_span(texts):
    res = []
    for text in texts:
        sentence_pieces = [i for i in re.split(r' *[\.\?,!\n][\'"\)\]]* *', text) if i != '']
        cur = 0
        tmp = []
        for i in sentence_pieces:
            idx = text.find(i, cur)
            if idx == -1:
                raise NotImplementedError
            else:
                cur = idx
                end = cur + len(i)
                tmp.append((cur, end))
                cur = end
        res.append(tmp)
    return res


def get_sentence_span(texts):
    res = []
    for text in texts:
        sentences = [i for i in re.split(r' *[\.\?!\n][\'"\)\]]* *', text) if i != '']
        cur = 0
        tmp = []
        for i in sentences:
            idx = text.find(i, cur)
            if idx == -1:
                raise NotImplementedError
            else:
                cur = idx
                end = cur + len(i)
                tmp.append((cur, end))
                cur = end
        res.append(tmp)
    return res


def get_paragraph_span(texts):
    res = []
    for text in texts:
        paragraphs = [i for i in re.split(r' *[\n][\'"\)\]]* *', text) if i != '']
        cur = 0
        tmp = []
        for i in paragraphs:
            idx = text.find(i, cur)
            if idx == -1:
                raise NotImplementedError
            else:
                cur = idx
                end = cur + len(i)
                tmp.append((cur, end))
                cur = end
        res.append(tmp)
    return res


def get_special_tokens():
    return {
        '\r\n': '[BR]',
        '\\u': '[SU]',
        'prompt_question': '[PQ]',
        'prompt_title': '[PT]',
        'prompt_text': '[ST]',
        'text': '[SM]'
    }


def prepare_folds(train_ds, args):

    train_ds['full_text'] = '[PQ] ' + train_ds['prompt_question'] + ' [PT] ' + train_ds['prompt_title'] + ' [ST] ' + \
                            train_ds['prompt_text'] + ' [SM] ' + train_ds['text']

    Fold = GroupKFold(n_splits=args.n_fold)
    for n, (train_index, val_index) in enumerate(
            Fold.split(train_ds[args.model['text']], train_ds[args.model['target']], train_ds['prompt_id'])):
        train_ds.loc[val_index, 'fold'] = int(n)

    train_ds['fold'] = train_ds['fold'].astype(int)

    return train_ds
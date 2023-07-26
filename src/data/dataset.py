import torch
import numpy as np
from torch.utils.data import Dataset
from data_utils import preprocess_text, get_span_from_text

from collections import defaultdict


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args['max_len']
        self.text_cols = args['text']
        self.target = args['target']
        self.span_cols = args['text']
        self.span = args['spans']
        df = self._preprocess(df)
        self.data = self._prepare_items(df)

    def _prepare_items(self, df):
        data = defaultdict(list)
        for k in self.text_cols:
            tokens = self.tokenizer(
                df[k],
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
                return_offsets_mapping=True if k in self.span_cols else False
            )
            for key in tokens:
                if key == 'token_type_ids': continue
                data[k + '_' + key] = tokens[key]
            for i in range(len(df[k + '_span_offset'])):
                span_labels = np.zeros(len(data[k + '_offset_mapping'][i])) - 1
                for idx, (sp, ep) in enumerate(df[k + '_span_offset'][i]):
                    for j, (s, e) in enumerate(data[k + '_offset_mapping'][i]):
                        if min(ep, e) - max(sp, s) > 0:
                            span_labels[j] = idx + 1
                data[k + '_span_label'].append(span_labels.tolist())
        for k in self.target:
            if not data['label']:
                data['label'] = [[i] for i in df[k]]
            else:
                data['label'] = [data['label'][i] + [df[k][i]] for i in range(len(df[k]))]
        return data

    def _preprocess(self, df):
        data = defaultdict(list)
        self.length = 0
        for col in df.columns:
            if col in self.text_cols:
                df[col] = df[col].astype(str).fillna('').apply(preprocess_text)
                data[col] += [text for text in df[col]]
                if col in self.span_cols and self.span:
                    data[col + '_span_offset'] += get_span_from_text(df[col], self.span)
                    data[col + '_span_count'] += [len(i) for i in data[col + '_span_offset']]
            elif col in self.target:
                data[col] += [i for i in df[col]]
            self.length = max(self.length, len(df[col]))
        return data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            k: self.data[k][idx] for k in self.data
        }


class Collatator:
    def __init__(self, args, tokenizer, inference=False):
        self.tokenizer = tokenizer
        self.num_targets = args['num_labels']
        self.inference = inference

    def __call__(self, batch):
        cols = batch[0].keys()
        out = defaultdict(list)
        for c in cols:
            if c == 'label' and self.inference or 'offset' in c: continue
            max_len = max([len(data[c]) for data in batch])
            pad_token = -1 if 'label' in c else (self.tokenizer.pad_token_id if 'ids' in c else 0)
            if c != 'label':
                out[c] = torch.tensor([data[c] + [pad_token] * (max_len - len(data[c])) for data in batch])
            else:
                out[c] = torch.tensor(
                    [data[c] + [[pad_token] * self.num_targets] * (max_len - len(data[c])) for data in batch])

        return out
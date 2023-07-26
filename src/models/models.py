import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint
import torch.nn.functional as F
from pooling import CustomPooling

import gc


class CommonLitModel(nn.Module):
    def __init__(self,
                 model,
                 pooling_params={},
                 use_gradient_checkpointing=False,
                 spans="sentence",
                 ):
        super().__init__()
        self.spans = spans

        self.backbone = model

        self.config = self.backbone.config
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features": self.config.hidden_size,
                                    "out_features": self.config.hidden_size
                                    })
        self.pool = CustomPooling(**self.pooling_params)

        if use_gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        #         self.wc_emb = WordCountEmbedding(self.config.hidden_size)

        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, num_layers=1, bidirectional=True,
                            batch_first=True)

        self.fc_wording = nn.Linear(4 * self.config.hidden_size, 1)
        self.fc_content = nn.Linear(2 * self.config.hidden_size, 1)
        self._init_weights(self.fc_wording)
        self._init_weights(self.fc_content)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _from_token_to_span(self, preds, labels_ids, attention_mask):
        TOK, SEQ = preds.shape
        predictions = []
        ids = torch.unique(labels_ids)
        for idx in ids:
            if idx != -1:
                mask = labels_ids == idx
                p = preds[mask].reshape(1, -1, SEQ)
                att = attention_mask[mask].reshape(1, -1)
                predictions.append(self.pool(p, att))
        return torch.cat(predictions)

    def from_token_to_span(self, preds, labels_ids, attention_mask):
        BS, _, SEQ = preds.shape
        if BS > 1:
            print('Span pooler only support batch size = 1')

        predictions = []
        for p, l, att in zip(preds, labels_ids, attention_mask):
            predictions.append(self._from_token_to_span(p, l, att))

        return torch.cat(predictions).reshape(BS, -1, SEQ)

    def forward(self, x):
        """
            x:
                prompt_text_input_ids,
                prompt_text_attention_mask,
                prompt_text_span_label,
                text_input_ids,
                text_attention_mask,
                text_span_label,
                prompt_instruction_input_ids,
                prompt_instruction_attention_mask
        """

        full_text = self.backbone(x['full_text_input_ids'], x['full_text_attention_mask']).last_hidden_state
        #         prompt_instruction = self.backbone(x['prompt_instruction_input_ids'], x['prompt_instruction_attention_mask']).last_hidden_state
        text = self.backbone(x['text_input_ids'], x['text_attention_mask']).last_hidden_state

        full_text = self.dropout(full_text)
        #         prompt_instruction = self.dropout(prompt_instruction)
        text = self.dropout(text)

        full_text = self.from_token_to_span(full_text, x['full_text_span_label'], x['full_text_attention_mask'])
        x['full_text_attention_mask'] = x['full_text_attention_mask'] * 0 + 1
        x['full_text_attention_mask'] = x['full_text_attention_mask'][:, full_text.size(1)]
        text = self.from_token_to_span(text, x['text_span_label'], x['text_attention_mask'])
        x['text_attention_mask'] = x['text_attention_mask'] * 0 + 1
        x['text_attention_mask'] = x['text_attention_mask'][:, text.size(1)]

        #         word_counts = torch.sum(x['text_attention_mask'], dim = -1)
        #         wc_emb = self.wc_emb(word_counts)
        #         full_text += wc_emb
        #         text += wc_emb
        bs, _, d = full_text.shape

        full_text = self.lstm(full_text)[1][0].view(-1, bs, 2 * d)
        text = self.lstm(text)[1][0].view(-1, bs, 2 * d)

        content = self.fc_content(full_text)
        wording = self.fc_wording(torch.cat((full_text, text), dim=-1))

        return torch.cat((content, wording), dim=-1)



import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from data import PeopleDaily
checkpoint=r"D:\LLM\LLM-main\Sequence Labeling\config"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
train_data = PeopleDaily('./example.train.txt')
valid_data = PeopleDaily('./example.dev.txt')
test_data = PeopleDaily('./example.test.txt')
id2label = {0:'O'}
for c in list(sorted(train_data.categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}


def collate_fn(batch_samples):
    batch_sentences, batch_tags = [], []
    for simple in batch_samples:
        batch_sentences.append(simple['sentence'])
        batch_tags.append(simple['labels'])
    batch_inputs=tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',

    )

    batch_label = np.full(batch_inputs['input_ids'].shape, -100, dtype=int)
    for s_idx, sentence in enumerate(batch_sentences):
        encoding = tokenizer(sentence, truncation=True)
        for char_start, char_end, _, tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            if token_start is None or token_end is None:
                continue
            batch_label[s_idx][token_start] = label2id[f"B-{tag}"]
            batch_label[s_idx][token_start + 1: token_end + 1] = label2id[f"I-{tag}"]
    return batch_inputs, torch.tensor(batch_label)



train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=collate_fn)

from torch import nn
import torch
from transformers import AutoConfig
from transformers import BertPreTrainedModel, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, len(id2label))
        self.post_init()

    def forward(self, x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
config = AutoConfig.from_pretrained(checkpoint)
model = BertForNER.from_pretrained(checkpoint, config=config).to(device)

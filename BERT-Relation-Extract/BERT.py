import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer


class MyDataset(Dataset):
    def __init__(self, data_type='train', max_length=128, bert_scale='base'):
        self.file = f'archive2/{data_type}.json'
        self.tokenizer = BertTokenizer.from_pretrained(f'bert_{bert_scale}_uncased')
        self.ent_markers = ['[e1s]', '[e1e]', '[e2s]', '[e2e]']
        self.tokenizer.add_tokens(self.ent_markers)
        self.vocab = self.tokenizer.get_vocab()

        with open(self.file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open('archive2/semeval_rel2id.json', 'r', encoding='utf-8') as f:
            self.rel_class_dict = json.load(f)

        self.cls_num = len(self.rel_class_dict)
        self.labels = np.eye(self.cls_num)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_dict = self.data[idx]
        token = item_dict['token']
        h, t = item_dict['h']['pos'], item_dict['t']['pos']

        token_list = (token[:h[0]] + [self.ent_markers[0]] + token[h[0]:h[1]] + [self.ent_markers[1]] + token[h[1]:t[0]]
                      + [self.ent_markers[2]] + token[t[0]:t[1]] + [self.ent_markers[3]] + token[t[1]:])
        tokenized = self.tokenizer(''.join(token_list), truncation=True,
                                   padding='max_length', max_length=self.max_length)

        ht_head = [self.tokenizer.get_vocab()['[e1s]'], self.tokenizer.get_vocab()['[e2s]']]
        ht_pos = [list(tokenized['input_ids']).index(em) for em in ht_head]
        rel_class = self.rel_class_dict[item_dict['relation']]
        rel_label = self.labels[rel_class].tolist()

        # print(ht_pos, list(tokenized['input_ids'])[ht_pos[0]], list(tokenized['input_ids'])[ht_pos[1]])
        # print(token_list['input_ids'])
        # print(token_list['attention_mask'])

        return {'ids': torch.tensor(tokenized['input_ids'], dtype=torch.long).squeeze(),
                'att_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long).squeeze(),
                'h': ht_pos[0],
                't': ht_pos[1],
                'rel_cls': torch.tensor(rel_label, dtype=torch.float32)}


class BertRelationExtract(nn.Module):
    def __init__(self, bert_scale='base'):
        super().__init__()
        with open('archive2/semeval_rel2id.json', 'r', encoding='utf-8') as f:
            self.rel_class_dict = json.load(f)

        self.bert_model = BertModel.from_pretrained(f'bert_{bert_scale}_uncased')
        self.fc = nn.Sequential(nn.BatchNorm1d(num_features=self.bert_model.config.hidden_size * 2),
                                nn.Linear(self.bert_model.config.hidden_size * 2, 768, bias=True),
                                nn.GELU(),
                                nn.BatchNorm1d(num_features=768),
                                nn.Linear(768, 19, bias=True),
                                nn.GELU())
        self.softmax = F.softmax
        # print(self.bert_model)

        for name, param in self.bert_model.named_parameters():
            for i in range(10):
                if f'encoder.layer.{i}' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, ids, att_mask, h, t):
        bert_output = self.bert_model(input_ids=ids, attention_mask=att_mask)
        bert_output = bert_output['last_hidden_state']

        fc_input = []
        for i in range(len(bert_output)):
            one_output = torch.concat((bert_output[i, h[i], :].unsqueeze(0),
                                       bert_output[i, t[i], :].unsqueeze(0)), dim=1)
            fc_input.append(one_output)

        fc_input = torch.concat(fc_input)
        fc_output = self.fc(fc_input)
        fc_output = self.softmax(fc_output, dim=1)
        return fc_output


if __name__ == "__main__":
    BERT_scale = 'base'     # 'base'或'large'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备：{device}")

    # 模型参数
    checkpoint_path = "best_pts/best.pt"
    train_epoches = 500
    lr = 0.001
    hidden_dim = 128
    max_acc = 0.0

    # 初始化模型，定义优化器
    model = BertRelationExtract(bert_scale=BERT_scale).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    train_dataset = MyDataset(data_type='train', max_length=hidden_dim, bert_scale=BERT_scale)
    val_dataset = MyDataset(data_type='test', max_length=hidden_dim, bert_scale=BERT_scale)
    print(f"Train Data Size: {len(train_dataset)}")
    print(f"Val Data Size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=10)

    if checkpoint_path != "":
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

    # 训练模型
    for epoch in range(train_epoches):
        stime = time.time()
        # 训练过程
        model.train()
        train_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            model.zero_grad()
            cls = model(data['ids'].to(device), data['att_mask'].to(device), data['h'], data['t'])

            loss = loss_func(cls, data['rel_cls'].to(device))
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        etime = time.time()
        torch.save(model.state_dict(), f"last(0.001lr).pt")

        if epoch % 1 == 0:
            avr_train_loss = train_loss / len(train_dataset)
            print(f"Epoch {epoch + 1}/{train_epoches} Train Loss: {avr_train_loss:.8f}, Time: {etime - stime}s")

        # 验证过程
        model.eval()
        stime = time.time()
        val_loss = 0.0
        max_f1 = 0.0

        true_label = []
        pred_label = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                model.zero_grad()
                cls = model(data['ids'].to(device), data['att_mask'].to(device), data['h'], data['t'])

                loss = loss_func(cls, data['rel_cls'].to(device))
                val_loss += loss.item()

                pred_label.append(int(cls.squeeze().argmax()))
                true_label.append(int(data['rel_cls'].squeeze().argmax()))
        etime = time.time()

        f1 = f1_score(true_label, pred_label, average='weighted')
        f1 = float(f1)

        if epoch % 1 == 0:
            avr_val_loss = val_loss / len(val_dataset)
            print(f"Epoch {epoch + 1}/{train_epoches} Val Loss: {avr_val_loss:.8f}, "
                  f"F1 Score: {f1:.8f}, Time: {etime - stime}s")

            if f1 > max_f1:
                max_f1 = f1
                torch.save(model.state_dict(), f"best.pt")

import json
import os
import re
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from BERT import BertRelationExtract


class MyDataset(Dataset):
    def __init__(self, rel_list, max_length=128, bert_scale='base'):
        self.tokenizer = BertTokenizer.from_pretrained(f'bert_{bert_scale}_uncased')
        self.ent_markers = ['[e1s]', '[e1e]', '[e2s]', '[e2e]']
        self.tokenizer.add_tokens(self.ent_markers)
        self.vocab = self.tokenizer.get_vocab()
        self.data = rel_list

        with open('archive2/semeval_rel2id.json', 'r', encoding='utf-8') as f:
            self.rel_class_dict: dict = json.load(f)
            self.idx_rel_class = {self.rel_class_dict[k]: k for k in self.rel_class_dict.keys()}

        self.cls_num = len(self.rel_class_dict)
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
        try:
            ht_pos = [list(tokenized['input_ids']).index(em) for em in ht_head]
        except:
            return dict()

        return {'ids': torch.tensor(tokenized['input_ids'], dtype=torch.long).squeeze(),
                'att_mask': torch.tensor(tokenized['attention_mask'], dtype=torch.long).squeeze(),
                'h': ht_pos[0], 'h_name': item_dict['h']['name'],
                't': ht_pos[1], 't_name': item_dict['t']['name']}


def text_process(text, entities):
    # 加载spaCy模型
    nlp = spacy.load("en_core_web_lg")
    text = text.strip()
    doc = nlp(text)

    # 对文本进行处理, 提取实体和关系
    ent_dict_list = []
    # print([ent for ent in doc.ents])
    for ent in doc.ents:
        if ent.text in entities:
            tokens = [token for token in ent]
            start, end = tokens[0].i, tokens[-1].i
            ent_dict_list.append({"name": ent.text, "pos": [start, end + 1]})

    # 列举关系对
    possible_rel = []
    for i in range(len(ent_dict_list)):
        e1 = ent_dict_list[i]
        for j in range(i + 1, len(ent_dict_list)):
            e2 = ent_dict_list[j]
            if e1["name"] != e2["name"]:
                possible_rel.append({"h": e1, "t": e2})

    for i in range(len(possible_rel)):
        possible_rel[i]['token'] = [t.text for t in doc]
        # print(possible_rel[i])

    return possible_rel


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertRelationExtract().to(device)
    state_dict = torch.load('best_pts/best.pt')
    model.load_state_dict(state_dict)

    # 只抽取感兴趣的实体
    entities = []
    with open('entities.txt', 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(line.strip())
    print(entities)

    # 需要做实体标注的原文本
    pred_file = 'inference_text/simpson/Summary.txt'
    filepath = 'inference_text/simpson'

    for file in os.listdir(filepath):
        pred_file = filepath + '/' + file
        print(pred_file)

        # 加载spaCy模型
        nlp = spacy.load("en_core_web_lg")
        with open(pred_file, 'r', encoding='utf-8') as f:
            text = f.read()
            # 删除引用中括号内容
            text = re.sub(r'\[(.*?)\]', '', text)
            # 删除ASCII控制字符
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', text)
            text = text.replace('\n', ' ')

        # 定义文本
        text = text.strip()
        doc = nlp(text)
        sentences = [s.text for s in doc.sents]  # 拆分句子

        output_file = 'predict_result' + pred_file[pred_file.rindex('/'):]
        if not os.path.exists('predict_result'):
            os.mkdir('predict_result')

        with open(output_file, 'w', encoding='utf-8') as f:
            for s in sentences:
                print(s)
                # print(s, file=f)
                predict = MyDataset(text_process(s, entities))
                loader = DataLoader(predict)

                # 验证过程
                model.eval()
                with torch.no_grad():
                    for i, data in enumerate(loader, 0):
                        if len(data) != 0:
                            cls = model(data['ids'].to(device), data['att_mask'].to(device), data['h'], data['t'])
                            cls_idx = int(cls.squeeze().argmax())

                            if cls_idx != predict.rel_class_dict['Other']:
                                ent_pair = data['h_name'][0] + ' ----- ' + data['t_name'][0]
                                pad_len = max(5, 50 - len(ent_pair))
                                print(ent_pair + ' ----- ' + predict.idx_rel_class[cls_idx])
                                print(ent_pair + ' ----- ' + predict.idx_rel_class[cls_idx], file=f)
                print()
                print(file=f)

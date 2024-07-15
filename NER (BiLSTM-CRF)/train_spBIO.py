import time
import numpy as np
import torch
import torch.nn as nn
from TorchCRF import CRF
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from gensim.models import Word2Vec
import gensim.downloader as api


class NerDataset(Dataset):
    def __init__(self, w2v_model):
        self.w2v = w2v_model
        self.sentences = []
        self.sentence_ents = []
        self.ent_type = {}

        with open('Sentences.txt', 'r', encoding='utf-8') as f1:
            for line in f1:
                if line[-1] == '\n':
                    # 导入句子，word2vec转向量
                    word_vecs = []
                    words = line.strip().split(' ')
                    for w in words:
                        try:
                            word_vecs.append(self.w2v.wv[w].tolist())
                        except:
                            word_vecs.append(np.zeros((self.w2v.vector_size,), dtype=np.float32).tolist())
                    self.sentences.append(word_vecs)
                else:
                    # 导入实体类别
                    typs = line.strip().split(' ')
                    typs.sort()
                    for i in range(len(typs)):
                        self.ent_type[typs[i]] = i

        with open('Entities.txt', 'r', encoding='utf-8') as f2:
            for line in f2:
                ent_codes = []
                ents = line.strip().split(' ')
                for ent in ents:
                    ent_codes.append(self.ent_type[ent])
                self.sentence_ents.append(ent_codes)

        # 检查句子数量是否匹配
        assert len(self.sentences) == len(self.sentence_ents), \
            f"Num of sentences({len(self.sentences)}) != Num of Entity Series({len(self.sentence_ents)})."

        # 检查句子中的单词数量是否匹配
        for i in range(len(self.sentences)):
            assert len(self.sentences[i]) == len(self.sentence_ents[i]), \
                (f"Num of word in sentence {i}({len(self.sentences[i])}) "
                 f"!= its len of entity list({len(self.sentence_ents[i])}).")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return (torch.tensor(self.sentences[idx]),
                torch.tensor(self.sentence_ents[idx]))


# 定义BiLSTM模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, hidden_dim=128, train_mode=True):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_dim = hidden_dim
        self.tag_to_idx = {'B-art': 0, 'B-eve': 1, 'B-geo': 2, 'B-gpe': 3, 'B-nat': 4, 'B-org': 5,
                           'B-per': 6, 'B-tim': 7, 'I-art': 8, 'I-eve': 9, 'I-geo': 10, 'I-gpe': 11,
                           'I-nat': 12, 'I-org': 13, 'I-per': 14, 'I-tim': 15, 'O': 16}
        self.idx_to_tag = {self.tag_to_idx[key]: key for key in self.tag_to_idx.keys()}
        self.tagset_size = len(self.tag_to_idx)
        # self.w2v = api.load('word2vec-google-news-300')
        self.w2v = Word2Vec.load('word2vec2.model')
        self.embedding_dim = self.w2v.vector_size

        if train_mode is True:
            self.dataset = NerDataset(self.w2v)
            train_size = int(0.8 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            print(f"Train Data Size: {train_size}")
            print(f"Val Data Size: {val_size}")

            self.train_data, self.val_data = torch.utils.data.random_split(self.dataset, [train_size, val_size])
            self.train_loader = DataLoader(self.train_data, batch_size=1, shuffle=True)
            self.val_loader = DataLoader(self.val_data, batch_size=1, shuffle=False)

        # self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2, num_layers=1, bidirectional=True)

        # LSTM的输出大小是hidden_dim，CRF的输入大小是tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层
        self.crf = CRF(self.tagset_size)

    def forward(self, sentence, tags=None):
        train = True if tags is not None else False
        lstm_out, _ = self.lstm(sentence.view(-1, 1, self.embedding_dim))
        lstm_out = lstm_out.view(sentence.shape[1], self.hidden_dim)
        emissions = self.hidden2tag(lstm_out).unsqueeze(0)

        if train is True:
            # CRF层的损失函数
            loss = -self.crf(emissions, tags, reduction='mean') * 10
            # 对于预测，使用维特比算法找到最佳路径
            pred_tags = self.crf.decode(emissions)
            return loss, pred_tags
        else:
            pred_tags = self.crf.decode(emissions)
            return [self.idx_to_tag[idx[0]] for idx in pred_tags]


if __name__ == "__main__":
    ckpt = ""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备：{device}")

    # 模型参数
    train_epoches = 100
    lr = 0.3
    max_acc = 0.0

    # 初始化模型，定义优化器
    model = BiLSTM_CRF().to(device)
    if ckpt is not None and ckpt != "":
        model.load(ckpt)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)

    # 训练模型
    for epoch in range(train_epoches):
        stime = time.time()
        # 训练过程
        model.train()
        train_loss = 0.0

        for i, train_data in enumerate(model.train_loader, 0):
            x, t = train_data
            model.zero_grad()

            loss, tags = model(x.to(device), t.to(device))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f"models/pretrained_wv/last1.pt")

        # 验证过程
        model.eval()
        val_loss = 0.0
        acc = 0.0

        for j, test_data in enumerate(model.val_loader, 0):
            x0, t0 = test_data
            loss, tags = model(x0.to(device), t0.to(device))
            val_loss += loss.item()

            acc += int(sum(sum(torch.tensor(tags).T == t0))) / int(t0.shape[1])

        etime = time.time()
        epoch_time = etime - stime

        if epoch % 1 == 0:
            avr_train_loss = train_loss / len(model.train_data)
            avr_val_loss = val_loss / len(model.val_data)
            avr_acc = acc / len(model.val_data)
            print(f"Epoch {epoch + 1} / {train_epoches} Train Loss: {avr_train_loss}, "
                  f"Test Loss: {avr_val_loss}, Accuracy: {avr_acc}, Epoch Time:{epoch_time:.3f}s")

            if avr_acc > max_acc:
                max_acc = avr_acc
                torch.save(model.state_dict(), f"models/pretrained_wv/best1.pt")

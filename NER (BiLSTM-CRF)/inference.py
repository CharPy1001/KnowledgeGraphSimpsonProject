import re
import spacy
import torch
import numpy as np
from gensim.models import Word2Vec
from train_BIO import BiLSTM_CRF

# 需要做实体标注的原文本
infer_text_file = 'inference_text/text2.txt'

# 加载spaCy模型
nlp = spacy.load("en_core_web_lg")
with open(infer_text_file, 'r', encoding='utf-8') as f:
    text = f.read()
    # 删除引用中括号内容
    text = re.sub(r'\[(.*?)\]', '', text)
    # 删除ASCII控制字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]', '', text)
    text = text.replace('\n', ' ')

# 定义文本
text = text.strip()
doc = nlp(text)
text_words_list = [token for token in doc]

"""for ent in doc.ents:
    print(ent.text)

    text_list = []
    tokens = [token for token in ent]
    for token in ent:
        text_list.append(text_words_list[token.i].text)
    print(' '.join(text_list), tokens[0].i, tokens[-1].i)

print('\n')"""

sentences = [s for s in doc.sents]  # 文章分句
sent_words = []                     # 所有句子的分词
sent_tokens = []                    # 分词后转词向量

w2v = Word2Vec.load('word2vec2.model')

for s in sentences:
    s_doc = nlp(s.text)

    # 获取句子中的单词
    words = [token.text for token in s_doc]
    words = [w for w in words if ' ' not in w]
    sent_words.append(words)

    word_vecs = []
    for w in words:
        try:
            word_vecs.append(w2v.wv[w].tolist())
        except:
            # 单个引号改为两个引号的词向量（和训练集保持一致）
            if w == '"':
                word_vecs.append(w2v.wv['""'].tolist())
            else:
                word_vecs.append(np.zeros((100,), dtype=np.float32).tolist())
    sent_tokens.append(word_vecs)

device = 'cuda'
model = BiLSTM_CRF(train_mode=False).to(device)
state_dict = torch.load("models/BIO_models/best.pt", map_location=device)
model.load_state_dict(state_dict)

# 保存推理结果
save_file = infer_text_file[:-4] + '_result.txt'
with open(save_file, 'w', encoding='utf-8') as f:
    for n in range(len(sent_tokens)):
        words = sent_words[n]
        pred = model(torch.tensor(sent_tokens[n]).unsqueeze(0).to(device))

        has_ent = False
        ent_words = []
        for i in range(len(words)):
            print(pred[i], words[i], file=f)

            if has_ent:
                if pred[i] == 'I':
                    ent_words.append(words[i])
                else:
                    has_ent = False
                    print(' '.join(ent_words))
                    ent_words.clear()
            else:
                if pred[i] == 'I':
                    has_ent = True
                    ent_words.append('*' + words[i])

            if pred[i] == 'B':
                has_ent = True
                ent_words.append(words[i])

        if len(ent_words) > 0:
            print(' '.join(ent_words))

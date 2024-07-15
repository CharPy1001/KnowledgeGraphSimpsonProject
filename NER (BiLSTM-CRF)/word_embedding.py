from gensim.models import Word2Vec

sentences = []
sentence_ents = []
ent_type = set()

with open('NER dataset.tsv', 'r', encoding='utf-8') as f1:
    s, ents = [], []
    for line in f1:
        try:
            word, typ, ent = line.replace(' ', '<spc>').strip().split('\t')
            if word == '<spc>':
                if typ == ent == '<spc>':
                    sentences.append(s)
                    sentence_ents.append(ents)
                    s, ents = [], []
                else:
                    print("Warning: Single space!")
            else:
                if word[0] == word[-1] == '"':
                    s.append(word[1:-1])
                else:
                    s.append(word)
                ents.append(ent)

            # 统计实体类别
            if ent not in ent_type:
                ent_type.add(ent)
        except:
            s.append('None')
            ents.append('O')

    sentences.append(s)
    sentence_ents.append(ents)

# 提取所有句子，保存到Sentences.txt
with open('Sentences.txt', 'w', encoding='utf-8') as f2:
    print(f"句子总数：{len(sentences)}")
    for s in sentences:
        f2.write(' '.join(s))
        f2.write('\n')
    f2.write(' '.join(ent_type - {'<spc>'}))

# 提取所有句子对应的实体类型，保存到Entities.txt
with open('Entities.txt', 'w', encoding='utf-8') as f3:
    print(f"实体标记句子总数：{len(sentence_ents)}")
    for ents in sentence_ents:
        f3.write(' '.join(ents))
        f3.write('\n')

model = Word2Vec(sentences, min_count=5, window=5, vector_size=100, workers=8)
model.save('word2vec2.model')

w2v = Word2Vec.load('word2vec2.model')
print(f'Word2Vec词汇表大小：{len(w2v.wv)}')

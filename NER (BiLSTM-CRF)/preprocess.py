import pandas as pd


# 打开原文件（二进制格式文件）以读取内容，转文本格式
dec_error = 0
lines = []
with open('NER dataset.csv', 'rb') as f1:
    for line in f1:
        try:
            lines.append(line.decode('utf-8'))
        except Exception as e:
            dec_error += 1
            print([line[i] for i in range(len(line))])
            # lines.append()

# 删除第一列不必要的信息
processed_lines = [lines[1][lines[1].index(',') + 1:]]
for line in lines[2:]:
    sep_index = line.index(',')
    if sep_index == 0:
        new_line = line[1:]
    else:
        new_line = ' , , \n' + line[sep_index + 1:]
    processed_lines.append(new_line)

# 初步处理后的数据写入NER dataset1.csv
data = ''.join(processed_lines)
with open('NER dataset1.csv', 'w', encoding='utf-8') as f2:
    f2.write(data)

# 将分隔符转换为'\t', 避免与数字中的逗号产生冲突
df = pd.read_csv('NER dataset1.csv', header=None, encoding='utf-8')
df.to_csv('NER dataset.tsv', sep='\t', encoding='utf-8', index=False, header=False)

with open('NER dataset.tsv', 'r', encoding='utf-8') as f3, open('./CRF++-0.57/train.data', 'w', encoding='utf-8') as f4:
    data = f3.read()
    data = data.replace(' \t \t ', '')
    data = data.replace('\t', ' ')
    print(data)
    f4.write(data)

with open('./CRF++-0.57/train.data', 'r', encoding='utf-8') as f5, open('./CRF++-0.57/tiny_train.data', 'w', encoding='utf-8') as f6:
    line_num = 199990
    while line_num > 0:
        line = f5.readline()
        f6.write(line)
        line_num -= 1

with open('./CRF++-0.57/train.data', 'r', encoding='utf-8') as f7, open('./CRF++-0.57/test.data', 'w', encoding='utf-8') as f8:
    line_num = 199991
    i = 0
    for line in f7:
        if i < line_num:
            i += 1
        else:
            f8.write(line)

print(f"Warning: Failed to decode {dec_error} word(s), which can't be decoded as a utf-8 format char(s).")
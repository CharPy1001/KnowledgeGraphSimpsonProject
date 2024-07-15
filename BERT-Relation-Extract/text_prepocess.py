file_folder = 'archive2'
file_list = ['semeval_train', 'semeval_val', 'semeval_test']
output_list = [f[f.index('_') + 1:] for f in file_list]

for i in range(len(file_list)):
    processed_lines = []
    with open(f'{file_folder}/{file_list[i]}.txt', 'r', encoding='utf-8') as f1:
        for line in f1:
            processed_lines.append('\t' + line[:-1] + ',\n')
        processed_lines[-1] = processed_lines[-1][:-2] + processed_lines[-1][-1]

    with open(f'{file_folder}/{output_list[i]}.json', 'w', encoding='utf-8') as f2:
        f2.write('[\n')
        f2.write(''.join(processed_lines))
        f2.write(']')

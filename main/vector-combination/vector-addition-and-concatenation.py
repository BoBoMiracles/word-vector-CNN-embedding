import numpy as np
import re

def load_word_vectors(filename, type):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as file:
        num_words, dim = map(int, file.readline().split())
        for line in file:
            parts = line.strip().split(' ')
            word = parts[0]
            if type == 1:  # 处理词向量文件
                if re.match('^[\u4e00-\u9fff]+$', word):  # 使用正则表达式匹配是否为汉字
                    vec = np.array(list(map(float, parts[1:])))
                    word_vectors[word] = vec
            if type == 2:  # 处理CNN字向量文件
                vec = np.array([float(element.strip('[,]')) for element in parts[1:]])
                # vec = np.array(list(map(float, parts[1][1:-1].split(','))))  # 去掉方括号，然后按逗号分隔解析向量
                word_vectors[word] = vec
    return word_vectors

def adjust_word_vector(word, word_vectors, char_vectors, type, a):
    word_vec = word_vectors[word]  # 获取整个词的向量
    char_vecs = [char_vectors[c] for c in word if c in char_vectors]  # 获取词中每个汉字的向量
    if char_vecs:
        char_mean = np.mean(char_vecs, axis=0)  # 计算汉字向量的均值
        if type == 1:
            return word_vec + a * char_mean  # 返回整个词向量加上汉字向量的均值
        elif type == 2:
            return np.concatenate((word_vec, char_mean))

def main():
    word_vectors_file = "word_embedding.txt"
    char_vectors_file = "output_CNN_vector_text_format.txt"
    output_file1 = "addition_word_vectors.txt"
    output_file2 = "concatenation_word_vectors.txt"

    word_vectors = load_word_vectors(word_vectors_file, 1)
    char_vectors = load_word_vectors(char_vectors_file, 2)

    addition_word_vectors = {}
    concatenation_word_vectors = {}
    for word in word_vectors:
        # type=1表示向量加法，type=2表示向量合并，a表示向量加法中字向量均值的影响程度，0-2
        addition_word_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, 1, 0.1)
        concatenation_word_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, 2, 1)

    with open(output_file1, 'w', encoding='utf-8') as file:
        file.write(f"{len(addition_word_vectors)} {len(addition_word_vectors[next(iter(addition_word_vectors))])}\n")
        for word, vec in addition_word_vectors.items():
            vec_str = ' '.join(str(x) for x in vec)
            file.write(f"{word} {vec_str}\n")

    with open(output_file2, 'w', encoding='utf-8') as file:
        file.write(f"{len(concatenation_word_vectors)} {len(concatenation_word_vectors[next(iter(concatenation_word_vectors))])}\n")
        for word, vec in concatenation_word_vectors.items():
            vec_str = ' '.join(str(x) for x in vec)
            file.write(f"{word} {vec_str}\n")

    # # 检验向量加法是否得到正确结果
    # a = word_vectors["为什么"]
    # b = (char_vectors["为"] + char_vectors["什"] + char_vectors["么"])/3
    # c = a + b
    # d = adjusted_word_vectors["为什么"]
    # print(c[50])
    # print(d[50])
    # if np.array_equal(c, d):
    #     print("向量完全相同")
    # else:
    #     print("向量不完全相同")


if __name__ == "__main__":
    main()
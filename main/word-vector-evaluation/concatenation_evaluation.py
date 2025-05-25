import numpy as np
import re
import os
import random
import pandas as pd


def load_word_vectors(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as file:
        num_words, dim = map(int, file.readline().split())
        for line in file:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array(list(map(float, parts[1:])))
            word_vectors[word] = vec
    return word_vectors


def adjust_word_vector(word, word_vectors, char_vectors):
    word_vec = word_vectors[word]  # 获取整个词的向量
    char_vecs = [char_vectors[c] for c in word if c in char_vectors]  # 获取词中每个汉字的向量
    if char_vecs:
        char_mean = np.mean(char_vecs, axis=0)  # 计算汉字向量的均值
        return np.concatenate((word_vec, char_mean))  # 将词向量和汉字向量拼接
    else:
        return word_vec


def read_vectors(vectors):
    iw = list(vectors.keys())
    wi = {word: idx for idx, word in enumerate(iw)}
    dim = len(next(iter(vectors.values())))
    matrix = np.zeros((len(iw), dim), dtype=np.float32)
    for i, word in enumerate(iw):
        matrix[i] = vectors[word]
    return vectors, iw, wi, dim


def read_analogy(path, iw, code='utf-8'):
    analogy = {}
    analogy_type = ""
    with open(path, encoding=code) as f:
        for line in f:
            oov = 0
            if line.strip().split()[0] == ':':
                analogy_type = line.strip().split()[1]
                analogy[analogy_type] = {}
                analogy[analogy_type]["questions"] = []
                analogy[analogy_type]["total"] = 0
                analogy[analogy_type]["seen"] = 0
                continue
            analogy_question = line.strip().split()
            for w in analogy_question[:3]:
                if w not in iw:
                    oov = 1
            if oov == 1:
                analogy[analogy_type]["total"] += 1
                continue
            analogy[analogy_type]["total"] += 1
            analogy[analogy_type]["seen"] += 1
            analogy[analogy_type]["questions"].append(analogy_question)

        for t in analogy:
            analogy[t]['iw'] = []
            analogy[t]['wi'] = {}
            for question in analogy[t]["questions"]:
                for w in question:
                    if w not in analogy[t]['iw']:
                        analogy[t]['iw'].append(w)
            for i, w in enumerate(analogy[t]['iw']):
                analogy[t]['wi'][w] = i
    return analogy


def normalize(matrix):
    norm = np.sqrt(np.sum(matrix * matrix, axis=1))
    matrix = matrix / norm[:, np.newaxis]
    return matrix


def guess(sims, analogy, analogy_type, iw, wi, word_a, word_b, word_c):
    sim_a = sims[analogy[analogy_type]["wi"][word_a]]
    sim_b = sims[analogy[analogy_type]["wi"][word_b]]
    sim_c = sims[analogy[analogy_type]["wi"][word_c]]

    add_sim = -sim_a + sim_b + sim_c
    add_sim[wi[word_a]] = 0
    add_sim[wi[word_b]] = 0
    add_sim[wi[word_c]] = 0
    guess_add = iw[np.nanargmax(add_sim)]

    mul_sim = sim_b * sim_c * np.reciprocal(sim_a + 0.01)
    mul_sim[wi[word_a]] = 0
    mul_sim[wi[word_b]] = 0
    mul_sim[wi[word_c]] = 0
    guess_mul = iw[np.nanargmax(mul_sim)]
    return guess_add, guess_mul


def evaluate_analogy(vectors, iw, wi, analogy):
    results = {}
    matrix = np.zeros((len(iw), len(vectors[iw[0]])), dtype=np.float32)  # 创建一个全零的矩阵
    for i, word in enumerate(iw):
        matrix[i, :] = vectors[word]  # 将每个词的向量填入矩阵
    matrix = normalize(matrix)  # 对矩阵进行归一化处理

    correct_add_total, correct_mul_total, seen_total = 0, 0, 0  # 初始化总体准确率统计量

    for analogy_type in analogy.keys():  # 遍历每种类比类型
        correct_add_num, correct_mul_num = 0, 0  # 初始化加法和乘法预测的正确计数
        analogy_matrix = matrix[
            [wi[w] if w in wi else random.randint(0, len(wi) - 1) for w in analogy[analogy_type]["iw"]]
        ]  # 创建类比矩阵
        sims = analogy_matrix.dot(matrix.T)  # 计算相似度
        sims = (sims + 1) / 2  # 将相似度转换为正数（用于乘法评估）
        for question in analogy[analogy_type]["questions"]:  # 遍历每个问题
            word_a, word_b, word_c, word_d = question
            guess_add, guess_mul = guess(sims, analogy, analogy_type, iw, wi, word_a, word_b, word_c)  # 获取预测结果
            if guess_add == word_d:
                correct_add_num += 1  # 如果加法预测正确，计数加1
            if guess_mul == word_d:
                correct_mul_num += 1  # 如果乘法预测正确，计数加1
        cov = float(analogy[analogy_type]["seen"]) / analogy[analogy_type]["total"]  # 计算覆盖率
        acc_add = float(correct_add_num) / analogy[analogy_type]["seen"] if analogy[analogy_type]["seen"] > 0 else 0  # 计算加法预测准确率
        acc_mul = float(correct_mul_num) / analogy[analogy_type]["seen"] if analogy[analogy_type]["seen"] > 0 else 0  # 计算乘法预测准确率
        results[analogy_type] = {
            "coverage": [cov, analogy[analogy_type]["seen"], analogy[analogy_type]["total"]],
            "accuracy_add": [acc_add, correct_add_num, analogy[analogy_type]["seen"]],
            "accuracy_mul": [acc_mul, correct_mul_num, analogy[analogy_type]["seen"]],
        }  # 存储结果

        # 统计总体准确率
        correct_add_total += correct_add_num
        correct_mul_total += correct_mul_num
        seen_total += analogy[analogy_type]["seen"]

    # 将总体准确率加入到结果中
    if seen_total == 0:
        results["total_accuracy"] = {
            "accuracy_add": 0.0,
            "accuracy_mul": 0.0
        }
    else:
        results["total_accuracy"] = {
            "accuracy_add": float(correct_add_total) / seen_total,
            "accuracy_mul": float(correct_mul_total) / seen_total
        }

    return results  # 返回最终结果


def save_results_to_excel(results, output_path):
    # 创建一个Excel writer对象
    writer = pd.ExcelWriter(output_path, engine='openpyxl')

    # 获取所有的维度和a值组合
    dimensions = sorted(set(dim for dim, _ in results.keys()))
    a_values = sorted(set(a for _, a in results.keys()))

    # 初始化字典来保存DataFrame
    data_add = {}
    data_mul = {}

    # 初始化数据结构
    for analogy_type in next(iter(results.values())).keys():
        data_add[analogy_type] = pd.DataFrame(index=[f"word_dim={dim}" for dim in dimensions],
                                              columns=[f"char_dim={a}" for a in a_values])
        data_mul[analogy_type] = pd.DataFrame(index=[f"word_dim={dim}" for dim in dimensions],
                                              columns=[f"char_dim={a}" for a in a_values])

    # 填充数据
    for (dim, a), result in results.items():
        for analogy_type, metrics in result.items():
            if analogy_type == "total_accuracy":
                data_add[analogy_type].loc[f"word_dim={dim}", f"char_dim={a}"] = metrics["accuracy_add"]
                data_mul[analogy_type].loc[f"word_dim={dim}", f"char_dim={a}"] = metrics["accuracy_mul"]
                continue
            data_add[analogy_type].loc[f"word_dim={dim}", f"char_dim={a}"] = metrics["accuracy_add"][0]
            data_mul[analogy_type].loc[f"word_dim={dim}", f"char_dim={a}"] = metrics["accuracy_mul"][0]

    # 将每个DataFrame写入Excel的不同sheet
    for analogy_type in data_add.keys():
        data_add[analogy_type].to_excel(writer, sheet_name=f"{analogy_type}_accuracy_add")
        data_mul[analogy_type].to_excel(writer, sheet_name=f"{analogy_type}_accuracy_mul")

    # 保存Excel文件
    writer._save()


def main():
    for a in [1, 2]:
        for b in [1, 2]:
            if a == 1:
                model = 'CBOW'
            elif a == 2:
                model = 'SG'
            if b == 1:
                test = 'semantic'
            elif b == 2:
                test = 'morphological'
            word_vectors_path = model + '_NS_wiki_txt'
            char_vectors_path = 'CNN_vector'
            analogy_path = 'testsets/CA8/' + test + '.txt'
            result_path = model + '_NS_concatenation_' + test + '_evaluation.xlsx'
            word_dims = [50, 80, 100, 200, 300, 500]
            char_dims = [50, 80, 100, 200, 300, 500]

            results = {}

            for word_dim in word_dims:
                word_vectors_file = f"{model}_NS_wiki_dim{word_dim}.txt"
                word_vectors = load_word_vectors(os.path.join(word_vectors_path, word_vectors_file))

                for char_dim in char_dims:
                    print(f"Evaluating {test} of model {model} for word_dim = {word_dim} with char_dim = {char_dim}")
                    char_vectors_file = f"output_CNN_vector_text_dim{char_dim}.txt"
                    char_vectors = load_word_vectors(os.path.join(char_vectors_path, char_vectors_file))

                    combined_vectors = {}
                    for word in word_vectors:
                        combined_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors)

                    vectors, iw, wi, dim = read_vectors(combined_vectors)
                    analogy = read_analogy(analogy_path, iw)

                    eval_results = evaluate_analogy(vectors, iw, wi, analogy)
                    results[(word_dim, char_dim)] = eval_results


                save_results_to_excel(results, result_path)


if __name__ == '__main__':
    main()

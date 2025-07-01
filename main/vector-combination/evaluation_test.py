import numpy as np
import re
import os
import random
import pandas as pd
import argparse


def load_word_vectors(filename):
    word_vectors = {}
    with open(filename, 'r', encoding='utf-8') as file:
        num_words, dim = map(int, file.readline().split())
        for line in file:
            parts = line.strip().split(' ')
            word = parts[0]
            if re.match('^[\u4e00-\u9fff]+$', word):  # 使用正则表达式匹配是否为汉字
                vec = np.array(list(map(float, parts[1:])))
                word_vectors[word] = vec
    return word_vectors


def adjust_word_vector(word, word_vectors, char_vectors, a):
    word_vec = word_vectors[word]  # 获取整个词的向量
    char_vecs = [char_vectors[c] for c in word if c in char_vectors]  # 获取词中每个汉字的向量
    if char_vecs:
        char_mean = np.mean(char_vecs, axis=0)  # 计算汉字向量的均值
        return word_vec + a * char_mean  # 返回整个词向量加上汉字向量的均值
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
        data_add[analogy_type] = pd.DataFrame(index=[f"dim={dim}" for dim in dimensions],
                                              columns=[f"a={a}" for a in a_values])
        data_mul[analogy_type] = pd.DataFrame(index=[f"dim={dim}" for dim in dimensions],
                                              columns=[f"a={a}" for a in a_values])

    # 填充数据
    for (dim, a), result in results.items():
        for analogy_type, metrics in result.items():
            if analogy_type == "total_accuracy":
                data_add[analogy_type].loc[f"dim={dim}", f"a={a}"] = metrics["accuracy_add"]
                data_mul[analogy_type].loc[f"dim={dim}", f"a={a}"] = metrics["accuracy_mul"]
                continue
            data_add[analogy_type].loc[f"dim={dim}", f"a={a}"] = metrics["accuracy_add"][0]
            data_mul[analogy_type].loc[f"dim={dim}", f"a={a}"] = metrics["accuracy_mul"][0]

    # 将每个DataFrame写入Excel的不同sheet
    for analogy_type in data_add.keys():
        data_add[analogy_type].to_excel(writer, sheet_name=f"{analogy_type}_accuracy_add")
        data_mul[analogy_type].to_excel(writer, sheet_name=f"{analogy_type}_accuracy_mul")

    # 保存Excel文件
    writer._save()

# def test_load_word_vectors():
#     word_vectors_path = 'CBOW_HS_wiki_txt/CBOW_HS_wiki_dim50.txt'
#     char_vectors_path = 'CNN_vector/output_CNN_vector_text_dim50.txt'
#     try:
#         word_vectors = load_word_vectors(word_vectors_path)
#         char_vectors = load_word_vectors(char_vectors_path)
#         print(f"Loaded {len(word_vectors)} word vectors from {word_vectors_path}")
#         print(f"Loaded {len(char_vectors)} char vectors from {char_vectors_path}")
#     except Exception as e:
#         print(f"Error loading vectors: {e}")
#
# test_load_word_vectors()
#
# def test_adjust_word_vector():
#     word_vectors_path = 'CBOW_HS_wiki_txt/CBOW_HS_wiki_dim50.txt'
#     char_vectors_path = 'CNN_vector/output_CNN_vector_text_dim50.txt'
#     try:
#         word_vectors = load_word_vectors(word_vectors_path)
#         char_vectors = load_word_vectors(char_vectors_path)
#         a = 0  # 可调整a值进行测试
#         adjusted_vectors = {}
#         for word in list(word_vectors.keys())[:10]:  # 仅打印前10个词向量
#             adjusted_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, a)
#             print(f"Adjusted vector for {word}: {adjusted_vectors[word]}")
#     except Exception as e:
#         print(f"Error adjusting vectors: {e}")
#
# test_adjust_word_vector()
#
# def test_read_analogy():
#     word_vectors_path = 'CBOW_HS_wiki_txt/CBOW_HS_wiki_dim50.txt'
#     try:
#         word_vectors = load_word_vectors(word_vectors_path)
#         _, iw, wi, _ = read_vectors(word_vectors)
#         analogy_path = 'testsets/CA8/semantic.txt'
#         analogy = read_analogy(analogy_path, iw)
#         print(f"Loaded analogy questions: {analogy.keys()}")
#         for key, value in analogy.items():
#             print(f"{key}: {value['questions'][:5]}")  # 打印前5个类比问题
#     except Exception as e:
#         print(f"Error reading analogy questions: {e}")
#
# test_read_analogy()
#
# def test_evaluate_analogy():
#     word_vectors_path = 'CBOW_HS_wiki_txt/CBOW_HS_wiki_dim50.txt'
#     char_vectors_path = 'CNN_vector/output_CNN_vector_text_dim50.txt'
#     try:
#         word_vectors = load_word_vectors(word_vectors_path)
#         char_vectors = load_word_vectors(char_vectors_path)
#         a = 0  # 可调整a值进行测试
#         adjusted_vectors = {}
#         for word in word_vectors:
#             adjusted_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, a)
#         vectors, iw, wi, _ = read_vectors(adjusted_vectors)
#         analogy_path = 'testsets/CA8/semantic.txt'
#         analogy = read_analogy(analogy_path, iw)
#         results = evaluate_analogy(vectors, iw, wi, analogy)
#         print(f"Evaluation results: {results}")
#     except Exception as e:
#         print(f"Error evaluating analogy: {e}")
#
# test_evaluate_analogy()
#
# def test_save_results():
#     results = {
#         (50, 0): {
#             'capital-common-countries': {
#                 'coverage': [1.0, 506, 506],
#                 'accuracy_add': [0.57, 289, 506],
#                 'accuracy_mul': [0.52, 263, 506]
#             }
#         }
#     }
#     dimensions = [50]
#     a_values = [0]
#     try:
#         save_results_to_excel(results, dimensions, a_values)
#         print("Results saved to evaluation_results.xlsx")
#     except Exception as e:
#         print(f"Error saving results: {e}")
#
# test_save_results()


def main():
    # for a in [2]:
    #     for b in [1, 2]:
    #         if a == 1:
    #             model = 'CBOW'
    #         elif a == 2:
    #             model = 'SG'
    #         if b == 1:
    #             test = 'semantic'
    #         elif b == 2:
    #             test = 'morphological'
    #         word_vectors_path = model + '_NS_wiki_txt'
    #         char_vectors_path = 'CNN_vector'
    #         analogy_path = 'testsets/CA8/' + test + '.txt'
    #         result_path = model + '_NS_addition_' + test + '_evaluation.xlsx'
    #         dimensions = [50, 80, 100, 200, 300, 500]
    #         a_values = [0, 0.5, 0.8, 1, 2, 5, 100]

    #         results = {}

    #         for dim in dimensions:
    #             word_vectors_file = f"{word_vectors_path}/{model}_NS_wiki_dim{dim}.txt"
    #             char_vectors_file = f"{char_vectors_path}/output_CNN_vector_text_dim{dim}.txt"

    #             word_vectors = load_word_vectors(word_vectors_file)
    #             char_vectors = load_word_vectors(char_vectors_file)

    #             for a in a_values:
    #                 print(f"Evaluating {test} of model {model} for dimension {dim} with a = {a}")
    #                 adjusted_vectors = {}
    #                 for word in word_vectors:
    #                     adjusted_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, a)

                    

    #         save_results_to_excel(results, result_path)
    
    for b in [1, 2]:
        if b == 1:
            test = 'semantic'
        elif b == 2:
            test = 'morphological'
        analogy_path = 'testsets/CA8/' + test + '.txt'

        adjusted_vectors = load_word_vectors('gated_vector_word_vectors.txt')
        vectors, iw, wi, _ = read_vectors(adjusted_vectors)
        analogy = read_analogy(analogy_path, iw)
        eval_results = evaluate_analogy(vectors, iw, wi, analogy)
        print("The result of " + test + "test:" )
        print(eval_results)
        print("---------------------------------------------------")


if __name__ == '__main__':
    main()
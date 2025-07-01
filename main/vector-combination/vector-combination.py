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
                word_vectors[word] = vec
    return word_vectors

def initialize_gate_parameters(d_word, d_char, fusion_type):
    """
    初始化门控机制所需的参数
    :param d_word: 词向量维度
    :param d_char: 字符向量维度
    :param fusion_type: 融合类型 (3: 向量级门控, 4: 特征交叉融合)
    :return: 参数字典
    """
    params = {}
    
    # 字符向量投影层参数（将字符向量投影到词向量空间）
    params['W_proj'] = np.random.randn(d_word, d_char) * 0.01
    params['b_proj'] = np.zeros(d_word)
    
    if fusion_type == 3:  # 向量级门控
        # 门控生成层参数
        params['W_gate'] = np.random.randn(d_word, 2 * d_word) * 0.01
        params['b_gate'] = np.zeros(d_word)
    
    elif fusion_type == 4:  # 特征交叉融合
        # 交叉特征层参数（拼接+交叉）
        params['W_fusion'] = np.random.randn(d_word, 3 * d_word) * 0.01
        params['b_fusion'] = np.zeros(d_word)
    
    return params

def gated_fusion(word_vec, char_mean, params, fusion_type):
    """
    门控机制融合向量
    :param word_vec: 词向量
    :param char_mean: 字符向量均值
    :param params: 门控参数
    :param fusion_type: 融合类型 (3: 向量级门控, 4: 特征交叉融合)
    :return: 融合后的向量
    """
    # 将字符向量投影到词向量空间
    char_proj = np.dot(params['W_proj'], char_mean) + params['b_proj']
    
    if fusion_type == 3:  # 向量级门控
        # 拼接词向量和字符向量
        combined = np.concatenate((word_vec, char_proj))
        
        # 生成门控向量 (sigmoid激活)
        gate = 1 / (1 + np.exp(-(np.dot(params['W_gate'], combined) + params['b_gate'])))
        
        # 应用门控机制
        fused_vec = gate * word_vec + (1 - gate) * char_proj
    
    elif fusion_type == 4:  # 特征交叉融合
        # 计算特征交叉项
        cross_term = word_vec * char_proj
        
        # 拼接所有特征
        combined = np.concatenate((word_vec, char_proj, cross_term))
        
        # 通过全连接层融合
        fused_vec = np.tanh(np.dot(params['W_fusion'], combined) + params['b_fusion'])
    
    return fused_vec

def adjust_word_vector(word, word_vectors, char_vectors, type, gate_params=None):
    word_vec = word_vectors[word]  # 获取整个词的向量
    char_vecs = [char_vectors[c] for c in word if c in char_vectors]  # 获取词中每个汉字的向量
    
    if not char_vecs:
        return word_vec  # 如果没有字符向量，返回原始词向量
    
    char_mean = np.mean(char_vecs, axis=0)  # 计算汉字向量的均值
    
    if type == 1:
        return word_vec + 0.1 * char_mean  # 返回整个词向量加上汉字向量的均值
    
    elif type == 2:
        return np.concatenate((word_vec, char_mean))
    
    elif type in [3, 4]:  # 门控融合
        return gated_fusion(word_vec, char_mean, gate_params, type)

def main():
    # 文件路径
    word_vectors_file = "word_embedding.txt"
    char_vectors_file = "output_CNN_vector_text_format.txt"
    
    # 加载词向量和字符向量
    word_vectors = load_word_vectors(word_vectors_file, 1)
    char_vectors = load_word_vectors(char_vectors_file, 2)
    
    # 获取向量维度
    d_word = len(next(iter(word_vectors.values())))
    d_char = len(next(iter(char_vectors.values())))
    
    # 初始化门控参数
    gate_params_vector = initialize_gate_parameters(d_word, d_char, 3)  # 向量级门控
    gate_params_cross = initialize_gate_parameters(d_word, d_char, 4)   # 特征交叉融合
    
    # 创建不同融合方法的字典
    addition_word_vectors = {}         # 加法融合
    concatenation_word_vectors = {}    # 拼接融合
    gated_vector_word_vectors = {}     # 向量级门控融合
    gated_cross_word_vectors = {}      # 特征交叉融合
    
    for word in word_vectors:
        # 基本融合方法
        addition_word_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, 1)
        concatenation_word_vectors[word] = adjust_word_vector(word, word_vectors, char_vectors, 2)
        
        # 门控融合方法
        gated_vector_word_vectors[word] = adjust_word_vector(
            word, word_vectors, char_vectors, 3, gate_params_vector)
        
        gated_cross_word_vectors[word] = adjust_word_vector(
            word, word_vectors, char_vectors, 4, gate_params_cross)
    
    # 保存加法融合结果
    with open("results/addition_word_vectors.txt", 'w', encoding='utf-8') as file:
        file.write(f"{len(addition_word_vectors)} {d_word}\n")
        for word, vec in addition_word_vectors.items():
            vec_str = ' '.join(f"{x:.6f}" for x in vec)
            file.write(f"{word} {vec_str}\n")
    
    # 保存拼接融合结果
    with open("results/concatenation_word_vectors.txt", 'w', encoding='utf-8') as file:
        new_dim = d_word + d_char
        file.write(f"{len(concatenation_word_vectors)} {new_dim}\n")
        for word, vec in concatenation_word_vectors.items():
            vec_str = ' '.join(f"{x:.6f}" for x in vec)
            file.write(f"{word} {vec_str}\n")
    
    # 保存向量级门控融合结果
    with open("results/gated_vector_word_vectors.txt", 'w', encoding='utf-8') as file:
        file.write(f"{len(gated_vector_word_vectors)} {d_word}\n")
        for word, vec in gated_vector_word_vectors.items():
            vec_str = ' '.join(f"{x:.6f}" for x in vec)
            file.write(f"{word} {vec_str}\n")
    
    # 保存特征交叉融合结果
    with open("results/gated_cross_word_vectors.txt", 'w', encoding='utf-8') as file:
        file.write(f"{len(gated_cross_word_vectors)} {d_word}\n")
        for word, vec in gated_cross_word_vectors.items():
            vec_str = ' '.join(f"{x:.6f}" for x in vec)
            file.write(f"{word} {vec_str}\n")
    
    # 保存门控参数（用于后续分析或微调）
    np.savez("results/gate_parameters.npz", 
             W_proj_vec=gate_params_vector['W_proj'],
             b_proj_vec=gate_params_vector['b_proj'],
             W_gate=gate_params_vector['W_gate'],
             b_gate=gate_params_vector['b_gate'],
             W_proj_cross=gate_params_cross['W_proj'],
             b_proj_cross=gate_params_cross['b_proj'],
             W_fusion=gate_params_cross['W_fusion'],
             b_fusion=gate_params_cross['b_fusion'])

if __name__ == "__main__":
    main()
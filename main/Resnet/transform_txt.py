def convert_to_text_format(file_path, output_path):
    # 读取txt文件，获取单词和向量列表
    with open(file_path, "r", encoding='utf-8') as file:
        lines = file.readlines()

    # 提取单词和向量
    words = []
    vectors = []
    for line in lines:
        parts = line.strip().split(" ")
        words.append(parts[0])
        vectors.append(" ".join(parts[1:]))

    # 获取单词数量和维度大小
    num_words = len(words)
    dimension_size = len(vectors[0].split(" "))

    # 将单词和向量写入新的txt文件
    with open(output_path, "w", encoding='utf-8') as file:
        # 写入元信息
        file.write(f"{num_words} {dimension_size}\n")

        # 写入单词和向量
        for word, vector in zip(words, vectors):
            file.write(f"{word} {vector}\n")




# 示例用法
# 将字典数据的txt文件转换成指定格式的txt文件
convert_to_text_format("output_CNN_vector.txt", "output_CNN_vector_text_format.txt")

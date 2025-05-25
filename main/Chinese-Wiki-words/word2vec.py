# coding:utf-8
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import logging
import os

if __name__ == "__main__":
    print('主程序开始执行...')

    input_file_name = 'wiki.txt'
    for a in [1, 2]:
        for b in [1, 2]:
            for i in [30, 40, 50]:  # [5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
                dim = 10*i
                if a == 0:
                    if b == 0:
                        folder_name = f"CBOW_NS_wiki"  # 文件夹名称
                        os.makedirs(folder_name, exist_ok=True)
                        model_file_name = f'CBOW_NS_wiki_dim{dim}.model'
                        file_path = os.path.join(folder_name, model_file_name)
                    elif b == 1:
                        folder_name = f"CBOW_HS_wiki"  # 文件夹名称
                        os.makedirs(folder_name, exist_ok=True)
                        model_file_name = f'CBOW_HS_wiki_dim{dim}.model'
                        file_path = os.path.join(folder_name, model_file_name)
                elif a == 1:
                    if b == 0:
                        folder_name = f"SG_NS_wiki"  # 文件夹名称
                        os.makedirs(folder_name, exist_ok=True)
                        model_file_name = f'SG_NS_wiki_dim{dim}.model'
                        file_path = os.path.join(folder_name, model_file_name)
                    elif b == 1:
                        folder_name = f"SG_HS_wiki"  # 文件夹名称
                        os.makedirs(folder_name, exist_ok=True)
                        model_file_name = f'SG_HS_wiki_dim{dim}.model'
                        file_path = os.path.join(folder_name, model_file_name)
                model_file_name = f'wiki_dim{dim}.model'

                print(f'dim{dim}转换过程开始...')

                logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  # 输出日志信息
                model = Word2Vec(LineSentence(input_file_name),
                                 vector_size=dim,
                                 alpha=0.025,
                                 window=5,
                                 min_count=3,
                                 sg=a,
                                 hs=b,
                                 workers=multiprocessing.cpu_count())
                print(f'dim{dim}转换过程结束！')

                print(f'dim{dim}开始保存模型...')
                model.save(file_path)
                print(f'dim{dim}模型保存结束！')

    print('主程序执行结束！')

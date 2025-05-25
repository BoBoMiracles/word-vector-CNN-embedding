from gensim.models import word2vec
import os

for a in [0, 1]:
    for b in [0, 1]:
        for i in [5, 6, 7, 8, 9, 10, 20, 30, 40, 50]:
            dim = 10 * i
            if a == 0:
                if b == 0:
                    txt_folder_name = f"CBOW_NS_wiki_txt"
                    txt_file_name = f'CBOW_NS_wiki_dim{dim}.txt'
                    model_folder_name = f"CBOW_NS_wiki"
                    model_file_name = f'CBOW_NS_wiki_dim{dim}.model'

                elif b == 1:
                    txt_folder_name = f"CBOW_HS_wiki_txt"
                    txt_file_name = f'CBOW_HS_wiki_dim{dim}.txt'
                    model_folder_name = f"CBOW_HS_wiki"
                    model_file_name = f'CBOW_HS_wiki_dim{dim}.model'

            elif a == 1:
                if b == 0:
                    txt_folder_name = f"SG_NS_wiki_txt"
                    txt_file_name = f'SG_NS_wiki_dim{dim}.txt'
                    model_folder_name = f"SG_NS_wiki"
                    model_file_name = f'SG_NS_wiki_dim{dim}.model'

                elif b == 1:
                    txt_folder_name = f"SG_HS_wiki_txt"
                    txt_file_name = f'SG_HS_wiki_dim{dim}.txt'
                    model_folder_name = f"SG_HS_wiki"
                    model_file_name = f'SG_HS_wiki_dim{dim}.model'

            os.makedirs(txt_folder_name, exist_ok=True)
            save_file_path = os.path.join(txt_folder_name, txt_file_name)
            load_file_path = os.path.join(model_folder_name, model_file_name)
            print(f'正在生成{save_file_path}')
            model = word2vec.Word2Vec.load(load_file_path)
            model.wv.save_word2vec_format(save_file_path)
            print(f'已生成{save_file_path}')

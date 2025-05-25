> ## 代码说明
>
> 本项目尝试实现基于中文字符图形文字的词向量生成和评估工作。利用ResNet50模型来实现中文字符图形信息的提取，并和Word2Vec基准模型的词向量结合，最终评估得到的词向量的效果。
>
> ### 1. 语料库加载和词向量提取
>
> 见<Chinese-Wiki-words>文件夹，利用<wiki-words-loading>对原始文本进行清洗和预处理后，“word-vec.py”包含的基准模型可以提取出不同维度的词向量，“model2txt.py”将模型中的词向量输出为txt格式。
>
> ### 2. Resnet50提取中文字符图形信息
>
> 见<ResNet>文件夹，使用“Noto-Sans-SC”字体文件，“train-font-data.py”文件用ResNet50模型将字体图片转化为字符向量，再依次经过“load-vector.py”和“transform-txt.py”的处理，得到中文字符向量的txt格式文件output-CNN-vector-text-format.txt”
>
> ### 3. 词向量和字形向量的结合
>
> 见<vector-combination>文件夹，用向量加法或者拼接的方式，将词向量和字形向量结合起来，得到最终的向量。
>
> ### 4. 最终的词向量的评估工作
>
> 见<word-vector-evaluation>文件夹，使用CA8测试集，评估词向量在语义学和形态学上的效果。其中“addition-evaluation.py”和“concentration-evaluation.py”两个文件，分别批量生成和评估了向量加法和向量拼接两种词向量的效果，可以直接导入基准词向量和字符图形向量进行评估，无须经过第三个步骤<vector-combination>的处理。

参考文献：

1. https://github.com/liuwenqiang1202

2. Shen Li, Zhe Zhao, Renfen Hu, Wensi Li, Tao Liu, Xiaoyong Du, <a href="http://aclweb.org/anthology/P18-2023"><em>Analogical Reasoning on Chinese Morphological and Semantic Relations</em></a>, ACL 2018.



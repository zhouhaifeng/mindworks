from gensim.models import Word2Vec
import nltk
#nltk.download('punkt')  # 下载必要的NLTK数据

class searchwords:
    def __init__(self):
        self.word2vec_model = None
        # 加载预训练的Word2Vec模型（示例使用的是Google的Word2Vec模型，您可以使用自己的模型）
        # 这里的'path/to/your/word2vec/model'应替换为您的Word2Vec模型的路径
        word2vec_model = Word2Vec.load('/word2vec/model')
        if word2vec_model == None:
            print("init word2vec_model error!")

        return
    def word2vec(self):
        # 输入语句
        input_sentence = ""

        # 将输入语句分词
        words = nltk.word_tokenize(input_sentence)

        # 初始化一个空列表来存储每个词的向量
        word_vectors = []

        # 遍历每个词并将其转换为向量
        for word in words:
            try:
                # 使用Word2Vec模型将词转换为向量
                vector = self.word2vec_model.wv[word]
                word_vectors.append(vector)
            except KeyError:
                # 如果词不在词汇表中，可以选择跳过它或使用一个默认向量
                # 在这个示例中，我们将跳过未知词汇
                pass

        # 打印每个词的向量
        for word, vector in zip(words, word_vectors):
            print(f"Word: {word}, Vector: {vector}")
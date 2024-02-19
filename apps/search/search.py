package search

from knowlage.grpahdb import db
import sentence

class Search:
    def __init__(self):
        self.input_sentence = None
        self.input_sentence = input()
        return

    def search(self):
        sentence = Sentence(self.input_sentence)
        if sentence = "":
            print("input sentence error!")
            return

        vectors = sentence.word2vec()
        if vectors == None:
            print("can not get vectors")
            return

        graphql_db = GraphQLDB()
        if graphql_db == None:
            print("graphql init error!")
            return

        result = graphql_db.search_vector(vectors)
        # 打印结果
        print(json.dumps(result.data, indent=2))


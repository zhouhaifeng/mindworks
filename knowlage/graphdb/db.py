from graphql import (
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
    GraphQLField,
    parse,
    execute,
)
import numpy as np
import json

#vectors = {
#    "vector1": np.array([1.0, 2.0, 3.0]),
#    "vector2": np.array([4.0, 5.0, 6.0]),
#}

class graphql:
    def __init__:
        self.client = None
        return
    def search_vector(vectors):
        # 定义一个类型来表示向量
        VectorType = GraphQLObjectType(
            name="Vector",
            fields=lambda: {
                "components": GraphQLField(GraphQLString),
                "magnitude": GraphQLField(GraphQLString),
            },
        )

        # 定义一个根查询类型
        QueryType = GraphQLObjectType(
            name="Query",
            fields={
                "vector": GraphQLField(
                    VectorType,
                    args={
                        "name": GraphQLField(GraphQLString),
                    },
                    resolve=lambda root, info, name: {
                        "components": json.dumps(list(vectors[name])),
                        "magnitude": str(np.linalg.norm(vectors[name])),
                    },
                ),
            },
        )

        # 创建GraphQL模式
        schema = GraphQLSchema(query=QueryType)

        # 处理GraphQL查询
        query_str = """
        {
        vector(name: "vector1") {
            components
            magnitude
        }
        }
        """

        query = parse(query_str)
        result = execute(schema, query)

        # 打印结果
        print(json.dumps(result.data, indent=2))
        return result

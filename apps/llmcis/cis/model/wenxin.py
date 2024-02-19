import requests
import json
#在此模块中调用大模型提供的api,实现单文档/多文档摘要生成
#调用方: flink, 根据处理后的文档生成摘要
#todo: baidu上下文规格太小,需要大模型+小模型/图数据库, 大模型处理通用信息, 小模型/图数据库处理行业信息
#todo: 不同模型使用不同的类



class WenxinModel():
    def __init__(self, ak, sk):
        self.ak = ak
        self.sk = sk

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.ak, "client_secret": self.sk}
        return str(requests.post(url, params=params).json().get("access_token"))

    #api to get summay of text from llm by calling api of url
    #input:
    #      url: api of url
    #      prompt: text
    #return:
    #      summary of text

    def api_get_summary(self, url, prompt):
        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "system":"你是一名高级文秘，仔细阅读下面的文章后给出这篇文章的summary，前10个关键词(keywords)，来源(source)，标题(title)，发布时间(public_time)等，结果必须用json代码的格式输出"
            })

        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)

        #todo: 错误处理

        # 字符串转化为 json
        response_json = json.loads(response.text)
        return(response_json)

    # 提示词给 LLM 返回答案
    # todo: api中未使用的字段, 错误处理
    def chat_with_llm_text(self, prompt):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + self.get_access_token() 
        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "system":"你是一名高级文秘，仔细阅读下面的文章后给出这篇文章的summary，前10个关键词(keywords)，来源(source)，标题(title)，发布时间(public_time)等，结果必须用json代码的格式输出"
            })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        # 字符串转化为 json
        response_json = json.loads(response.text)
        return(response_json)

    # todo: api中未使用的字段, 错误处理
    #todo: 将url的配置分离出去
    #todo: 将调用llm api分离出去
    def chat_with_llm_multitext(self, prompt):
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + self.get_access_token() 
        payload = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": str(prompt)
                    }
                ],
                "system":"你是一名高级文秘，仔细阅读下面的文章后给出这篇文章的summary，前10个关键词(keywords)，来源(source)，标题(title)，发布时间(public_time)等，结果必须用json代码的格式输出"
            })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response.text)
        # 字符串转化为 json
        response_json = json.loads(response.text)
        return(response_json)

    def gen_text_summary(self, text):
        #chat with llm to get summary of the text
        response_json = self.chat_with_llm_text(text)
        print(response_json)

        if response_json == None:
            print("gen_text_summary: chat_with_llm_text return none:\r\n")
            return

        if response_json["result"] == None:
            print("gen_text_summary: there is no result field\r\n")
            return

        #todo: get result of response
        return response_json["result"]

    def gen_multitext_summary(self, text_list):
        response_json = self.chat_with_llm_multitext(text_list)
        if response_json == None:
            print("gen_multitext_summary: chat_with_llm_multitext return none:\r\n")
            return

        if response_json["result"] == None:
            print("gen_multitext_summary: there is no result field\r\n")
            return

        #return result field in response
        print(response_json)
        return response_json["result"]


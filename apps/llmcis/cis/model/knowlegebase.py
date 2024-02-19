import requests
import json

AK = "093oQcLLC5zh1FOrH12iBuVO"
SK = "nUIBkeLGcnSV0F74jDpy2UqbNGj7utoc"

class BaiduKnowlegebasePlugin():
    def __init__(self, ak, sk, plugin_id):
        self.ak = ak
        self.sk = sk
        self.plugin_id = plugin_id
        #self.kb_name = kb_name

    def get_access_token(self):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=" + self.ak + "&client_secret=" + self.sk
        
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def search(self, prompt_words):

        #chat_completion = qianfan.ChatCompletion(ak="API Key", sk="Secret Key")
        # Plugin 知识库展示
        #endpoint_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/plugin/llmcis/"
        #plugin = qianfan.Plugin(ak = AK, sk = SK, endpoint = endpoint_url)
        #response = plugin.do(plugin = "uuid-zhishiku", prompt = prompt_words)

        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/plugin/" + self.plugin_id + "/?access_token=" + self.get_access_token()
        
        payload = json.dumps({
            "query": prompt_words,
            "plugins":["uuid-zhishiku"],
            "verbose":True,
            "stream": True
        })
        headers = {
            'Content-Type': 'application/json'
        }
        
        #print([url, payload, headers])
        response = requests.request("POST", url, headers=headers, data=payload, stream=True)

        for line in response.iter_lines():
            print(line)
        

if __name__ == '__main__':
    #ak, sk, plugin_id
    baidu_kb = BaiduKnowlegebasePlugin(AK, SK, "fv320eb9tj5dm4ib")
    response = baidu_kb.search("10月份螺纹钢的价格走势是怎样的?")
    print(response)
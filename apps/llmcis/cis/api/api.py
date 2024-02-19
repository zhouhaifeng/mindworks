#!/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask
from model import wenxin
from model import knowlegebase

AK = "093oQcLLC5zh1FOrH12iBuVO"
SK = "nUIBkeLGcnSV0F74jDpy2UqbNGj7utoc"

if __name__ == '__main__':
    app = Flask(__name__)
    @app.route('/')
    def index():
        return jsonify({'name': 'llmcis',
                        'email': 'higgsai@outlook.com'})

    @app.route('/report', methods=['POST'])
    def report():
        model = WenxinModel()
        if model == None:
            return jsonify({'error_message': 'report api can not get model',
                    'error_code': '10001'})     

        #todo: using fields in request
        text = json.loads(request.data)
        if text == None:
            return jsonify({'error_message': 'report api can not get request data',
                    'error_code': '10002'})    

        summary = wenxin.gen_text_summary(text)
        return jsonify({'name': 'llmcis',
                        'report': summary})

    @app.route('/search')
    def search():
        #todo: 目前需在代码中写明AK, SK
        plugin = BaiduKnowlegebasePlugin(AK, SK, "fv320eb9tj5dm4ib")
        if plugin == None:
            return jsonify({'error_message': 'search api can not get plugin',
                    'error_code': '10003'})  

        #search words
        #todo: using fields in request
        text = json.loads(request.data)
        if text == None:
            return jsonify({'error_message': 'report api can not get request data',
                    'error_code': '10004'})   

        result = plugin.search(text)
        if result == None:
            return jsonify({'error_message': 'search api can not get result',
                    'error_code': '10005'})  

        return jsonify({'name': 'llmcis',
                        'result': result})

    app.run()

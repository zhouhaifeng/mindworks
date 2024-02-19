import scrapy

class SteelSpider(scrapy.Spider):
    name = 'steelspider'
    start_urls = ['https://news.zhaogang.com/']

    def parse(self, response):
        for news in response.xpath('//a[contains(@href, "news")]/@href').getall():
            yield {'news': news.get()}
            print(news.get())

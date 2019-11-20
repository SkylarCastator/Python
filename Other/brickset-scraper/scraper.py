import scrapy

class BrickSetSpider(scrapy.Spider):
    name = "brickset spider"
    start_urls = ['http://brickset.com/set/year-2016']

    def parse(self, response):
        SET_SELECTOR = '.set'
        for brickset in response.css(SET_SELECTOR):
            NAME_SELECTOR = 'h1 ::text'
            yield {'name': brickset.css(NAME_SELECTOR).extract_first()}

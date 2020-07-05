##
# 一个爬虫，用于在百度百科搜索词条，返回网页内容（以防百度封IP，每秒爬一次）
##
####################
import re
import requests
from bs4 import BeautifulSoup


class webCrawlerBaiduBaike(object):
    """docstring for webCrawlerBaiduBaike"""

    def __init__(self, url='http://baike.baidu.com/search/word'):
        super(webCrawlerBaiduBaike, self).__init__()
        self.url = url

    # input keyword
    # output resquests object
    # url = 'http://baike.baidu.com/search/word'
    def search(self, searchItem):
        word = self.getWordClean(searchItem)
        res = requests.get(self.url, params={'word': word})
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text)

        # 查找不到词条，那么对返回的搜索列表进行二次查询
        if res.url[:35] == "http://baike.baidu.com/search/none?":
            for soup in self.findInSuggestList(word, soup):
                yield soup

        # 直接进入词条
        else:
            # isDrug, drugCharacterIncluded = findCharacterFromHtml(soup)
            yield soup

    def getWordClean(self, word):

        word = re.sub("\((.*)\)", "", word)
        word = word.replace(u"★", "")
        return word

    # 直接查询不到条目时，百度会返回推荐条目列表。本程序会在推荐条目列表中尝试搜索合适的条目
    def findInSuggestList(self, word, soup):

        linkList = soup.find_all('a', class_='result-title')

        for link in linkList:
            ajdustedItemName = link.text.replace('_百度百科', "")
            iPos = word.find(ajdustedItemName)
            # 找到最可能的匹配条目
            if iPos != -1:
                # 获得该条目的链接
                tmpUrl = link['href']
                res = requests.get(tmpUrl)
                res.encoding = 'utf-8'
                soup = BeautifulSoup(res.text)
                yield soup


# 使用方法 | example for using
if __name__ == '__main__':
    import requests

    url = 'https://rest.coinapi.io/v1/exchangerate/BTC/USD'
    headers = {'X-CoinAPI-Key': '73034021-THIS-IS-SAMPLE-KEY'}
    response = requests.get(url, headers=headers)

    print(response)



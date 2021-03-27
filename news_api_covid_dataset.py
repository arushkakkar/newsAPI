import newsapi
from newsapi.newsapi_client import NewsApiClient
import spacy
import en_core_web_lg
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from string import punctuation

def get_keywords_eng(content, nlp_eng):
    result = []
    doc = nlp_eng(content)
    for token in doc:
        if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation or token.text in ['LiveUpdated']):
            continue
        if (token.pos_ in ['NOUN', 'VERB', 'PROPN']):
            result.append(token.text)
    
    return result

def main():
    nlp_eng = en_core_web_lg.load()
    newsapi = NewsApiClient("5e6cbadeb56e4625a5955a226a52935a")
    dados = []

    for i in range(1, 6):
        articles = newsapi.get_everything(q = 'coronavirus', language = 'en', from_param = '2021-02-27', to = '2021-03-26', sort_by = 'relevancy', page = i)
        for article in articles['articles']:
            title = article['title']
            date = article['publishedAt']
            description = article['description']
            content = article['content']
            dados.append({'title':title, 'date':date, 'desc':description, 'content':content})
    
    df = pd.DataFrame(dados)
    df = df.dropna()
    df.head()
    
    results = []

    for content in df.content.values:
        results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content, nlp_eng)).most_common(5)])
    
    df['keywords'] = results

    print(df)

    text = str(results)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    df.to_csv("/home/arush/dataset.csv", index = False)


if __name__ == '__main__':
    main()

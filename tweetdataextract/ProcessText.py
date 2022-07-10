import re

# commands to install textblob are at location https://pypi.org/project/textblob/
from textblob import TextBlob

# Installation steps for wordCloud https://pypi.org/project/wordcloud/
# wordcloud needs visual tools installed on the machine https://visualstudio.microsoft.com/visual-cpp-build-tools/
# from wordcloud import WordCloud

stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]


# def cleanText(text):
#     text = re.sub(r'@[A-Za-z0-9]+', '', text) #Removes @ mentions
#     text = re.sub(r'#', '', text) #Removes @ mentions
#     text = re.sub(r'RT[\s]+', '', text)
#     text = re.sub(r'https?:\/\/\S+', '', text)
#     text = re.sub(r'https?:\/n', '', text)
#     text = text.replace("\n", "") 
#     text = text.replace(b"\xe2\x82\xb977".decode("utf-8"), '' )
#     text = text.encode('ascii', errors='ignore').decode("utf-8")
#     return text

def clean_tweet(tweet):
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
    
    
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
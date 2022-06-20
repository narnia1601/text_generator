import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('npr.csv')

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')

dtm = cv.fit_transform(df['Article'])

lda = LatentDirichletAllocation(n_components=7, random_state=42)

lda.fit(dtm)

single_topic = lda.components_[0]

top_twenty_words = single_topic.argsort()[-20:]

topic_results = lda.transform(dtm)

df['Topic'] = topic_results.argmax(axis=1)

print(df)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
nltk.download('wordnet')

from app import df_comments as df

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# remove html tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# remove emoji
def clean_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

df2 = df[['author', 'text']]

df['text'] = df['text'].apply(clean_emoji)
df['text'] = df['text'].apply(clean_html)
df['text'] = df['text'].apply(clean_text)

df = df[['author', 'text']]

df = df.dropna(subset=['text'])
df = df[df['text'] != '']
df = df.reset_index(drop=True)

# Load Tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load Model
with open('samarth.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict Sentiment
def predict_sentiment(review):
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

df['sentiment'] = df['text'].apply(predict_sentiment)

df_positive = df[df['sentiment'] == 'positive']
df_negative = df[df['sentiment'] == 'negative']

df_positive['text'] = df_positive['author'].map(df2.set_index('author')['text'])
df_negative['text'] = df_negative['author'].map(df2.set_index('author')['text'])


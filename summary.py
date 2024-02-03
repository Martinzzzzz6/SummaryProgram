import nltk
import os
import multiprocessing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag
from langdetect import detect
from functools import partial
from gensim import corpora, models
from textblob import TextBlob
from langdetect import detect
import string
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

def preprocess_text(text, language):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Extend stopwords list with common but less informative words
    extended_stopwords = set(stopwords.words(language)) if language in stopwords.fileids() else set()
    extended_stopwords.update(["the", "to", "a", "of", "in", "and", "is", "for", "on", "with", "that", "this", "it"])

    lemmatized_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        lemmatized = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]
        filtered = [word for word in lemmatized if word.lower() not in extended_stopwords]
        lemmatized_sentences.append(' '.join(filtered))

    return sentences, lemmatized_sentences

    
def score_sentence(sentence, important_words):
    word_count = len([word for word in word_tokenize(sentence.lower()) if word.isalpha()])
    word_score = sum(1 for word in word_tokenize(sentence.lower()) if word in important_words)
    return sentence, word_score / max(word_count, 1)

def summarize(text, mode='medium', language='english'):
    sentences, important_words = preprocess_text(text, language)
    pool = multiprocessing.Pool()
    score_partial = partial(score_sentence, important_words=important_words)
    sentence_scores = dict(pool.map(score_partial, sentences))
    pool.close()
    pool.join()

    summary_length = {'short': 0.25, 'medium': 0.5, 'long': 0.75}
    num_sentences = int(len(sentences) * summary_length.get(mode, 0.5))

    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary = " ".join(ranked_sentences[:num_sentences])
    return summary


def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment

def topic_modeling(text, num_topics=5, num_words=3):
    # Tokenize and clean text
    tokenized_text = [word_tokenize(doc.lower()) for doc in sent_tokenize(text)]
    tokenized_text = [[word for word in doc if word.isalpha() and word not in string.punctuation] for doc in tokenized_text]

    dictionary = corpora.Dictionary(tokenized_text)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_text]

    # Apply LDA model
    lda = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda.print_topics(num_words=num_words)
    return topics


def plot_word_frequency(text, num_words=10, file_name='word_frequency_histogram.png'):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    frequency = Counter(words)
    most_common = frequency.most_common(num_words)

    words, counts = zip(*most_common)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title("Word Frequency Histogram")
    plt.xlabel("Words")
    plt.ylabel("Frequency")

    # Save plot to a file instead of showing it
    plt.savefig(file_name)
    print(f"Histogram saved as {file_name}")

def save_summary_to_file(summary, default_filename):
    while True:
        user_input = input("Do you want to save the summary to a file? (y/n): ").strip().lower()
        if user_input in ['y', 'yes']:
            filename = input(f"Enter filename (default: {default_filename}): ").strip()
            filename = filename if filename else default_filename

            with open(filename, 'w', encoding='utf-8') as file:
                file.write(summary)
            print(f"Summary saved to {filename}")
            break
        elif user_input in ['n', 'no', '']:
            print("Summary not saved.")
            break

if __name__ == "__main__":
    with open('input.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    detected_language = detect(text)
    print(f"Detected Language: {detected_language}")

    valid_modes = ['short', 'medium', 'long']
    mode = input("Enter the summary mode (short, medium, long): ").lower()

    if mode not in valid_modes:
        print(f"Invalid mode selected. Defaulting to 'medium'.")
        mode = 'medium'

    summary = summarize(text, mode=mode, language=detected_language)
    print("\nSummary:\n", summary)

    # Sentiment Analysis
    sentiment = sentiment_analysis(summary)
    print("\nSentiment Analysis:")
    print("Polarity: ", sentiment.polarity)
    print("Subjectivity: ", sentiment.subjectivity)

    # Print summary and statistics
    original_len = len(text)
    summary_len = len(summary)
    reduction_percentage = ((original_len - summary_len) / original_len) * 100

    print(f"\nOriginal Text Length: {original_len} characters")
    print(f"Summary Length: {summary_len} characters")
    print(f"Reduction: {reduction_percentage:.2f}%")
    
    plot_word_frequency(text)

    # Save summary to a file
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"summary_{current_time}.txt"
    save_summary_to_file(summary, default_filename)

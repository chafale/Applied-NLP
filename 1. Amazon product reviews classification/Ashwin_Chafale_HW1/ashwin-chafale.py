import pandas as pd
import numpy as np
import nltk
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import contractions
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import warnings

warnings.filterwarnings('ignore')


def sampling(df):
    df['star_rating'] = df['star_rating'].astype(int)
    sample_size = 20000

    # five_star = df.loc[df['star_rating'] == 5].sample(sample_size)
    # four_star = df.loc[df['star_rating'] == 4].sample(sample_size)
    # three_star = df.loc[df['star_rating'] == 3].sample(sample_size)
    # two_star = df.loc[df['star_rating'] == 2].sample(sample_size)
    # one_star = df.loc[df['star_rating'] == 1].sample(sample_size)

    # To produce same prediction every time we run the code I have samples first 20,000 data from each class
    # To do random sampling we can un-comment the above code
    five_star = df.loc[df['star_rating'] == 5][:sample_size]
    four_star = df.loc[df['star_rating'] == 4][:sample_size]
    three_star = df.loc[df['star_rating'] == 3][:sample_size]
    two_star = df.loc[df['star_rating'] == 2][:sample_size]
    one_star = df.loc[df['star_rating'] == 1][:sample_size]

    return pd.concat([five_star, four_star, three_star, two_star, one_star], axis=0)


def data_cleaning(data):
    # convert all reviews to lower case
    data["pre_processed_reviews"] = data['review_body'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))

    # remove HTML tags as well as URLs from reviews.
    data["pre_processed_reviews"] = data["pre_processed_reviews"].apply(lambda x: BeautifulSoup(x).get_text())
    data["pre_processed_reviews"] = data["pre_processed_reviews"].apply(lambda x: re.sub(r"http\S+", "", x))

    # contractions
    data["pre_processed_reviews"] = data["pre_processed_reviews"].apply(lambda x: contractions.fix(x))

    # remove the non-alpha characters
    data["pre_processed_reviews"] = data["pre_processed_reviews"].apply(
        lambda x: " ".join([re.sub("[^A-Za-z]+", "", x) for x in nltk.word_tokenize(x)]))

    # remove extra spaces among the words
    data['pre_processed_reviews'] = data['pre_processed_reviews'].apply(lambda x: re.sub(' +', ' ', x))


def data_preprocessing(data):
    # lemmatization using wordnet lemmatizer
    lemmatizer = WordNetLemmatizer()
    data['pre_processed_reviews'] = data['pre_processed_reviews'].apply(
        lambda x: " ".join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))


def training_testing_split(data):
    five_star_X_train, five_star_X_test, five_star_Y_train, five_star_Y_test = train_test_split(
        data[data["star_rating"] == 5]["pre_processed_reviews"], data[data["star_rating"] == 5]["star_rating"], test_size=0.2,
        random_state=30)

    four_star_X_train, four_star_X_test, four_star_Y_train, four_star_Y_test = train_test_split(
        data[data["star_rating"] == 4]["pre_processed_reviews"], data[data["star_rating"] == 4]["star_rating"], test_size=0.2,
        random_state=30)

    three_star_X_train, three_star_X_test, three_star_Y_train, three_star_Y_test = train_test_split(
        data[data["star_rating"] == 3]["pre_processed_reviews"], data[data["star_rating"] == 3]["star_rating"], test_size=0.2,
        random_state=30)

    two_star_X_train, two_star_X_test, two_star_Y_train, two_star_Y_test = train_test_split(
        data[data["star_rating"] == 2]["pre_processed_reviews"], data[data["star_rating"] == 2]["star_rating"], test_size=0.2,
        random_state=30)

    one_star_X_train, one_star_X_test, one_star_Y_train, one_star_Y_test = train_test_split(
        data[data["star_rating"] == 1]["pre_processed_reviews"], data[data["star_rating"] == 1]["star_rating"], test_size=0.2,
        random_state=30)

    X_train = pd.concat([five_star_X_train, four_star_X_train, three_star_X_train, two_star_X_train, one_star_X_train])
    X_test = pd.concat([five_star_X_test, four_star_X_test, three_star_X_test, two_star_X_test, one_star_X_test])
    Y_train = pd.concat([five_star_Y_train, four_star_Y_train, three_star_Y_train, two_star_Y_train, one_star_Y_train])
    Y_test = pd.concat([five_star_Y_test, four_star_Y_test, three_star_Y_test, two_star_Y_test, one_star_Y_test])

    return X_train, X_test, Y_train, Y_test


def tf_idf(train, test):
    tf_idf_vector = TfidfVectorizer()
    tf_x_train = tf_idf_vector.fit_transform(train)
    tf_x_test = tf_idf_vector.transform(test)
    return tf_x_train, tf_x_test


class AmazonReviewsSentimentAnalysis:
    def __init__(self):
        # Reading the data
        self.df = pd.read_csv("amazon_reviews_us_Jewelry_v1_00.tsv", sep='\t', header=0, on_bad_lines='skip')
        self.df = self.df[['review_body', 'star_rating']]
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        # Data Sampling
        self.data = sampling(self.df)

        # self.avg_review_length_pre_cleaning = round(self.data['review_body'].str.len().mean(), 2)

        # Data Cleaning
        data_cleaning(self.data)

        # self.avg_review_length_post_cleaning = round(self.data['pre_processed_reviews'].str.len().mean(), 2)

        # Data Pre-processing
        data_preprocessing(self.data)

        # self.avg_review_length_post_preprocessing = 78.67

        # print(f"{self.avg_review_length_pre_cleaning}, {self.avg_review_length_post_cleaning}")
        # print(f"{self.avg_review_length_post_cleaning}, {self.avg_review_length_post_preprocessing}")
        # print()

        # train - test split
        self.X_train, self.X_test, self.Y_train, self.Y_test = training_testing_split(self.data)

        # tf_idf
        self.tf_X_train, self.tf_X_test = tf_idf(self.X_train, self.X_test)

        self.cols = ['1', '2', '3', '4', '5', 'weighted avg']
        self.metrics = ['precision', 'recall', 'f1-score']

        # ML Models
        self.perceptron()
        self.svm()
        self.logistic_regression()
        self.multinomial_nb()

    def perceptron(self):
        perceptron = Perceptron(max_iter=1000, random_state=0)
        perceptron.fit(self.tf_X_train, self.Y_train)
        y_test_predicted = perceptron.predict(self.tf_X_test)

        report = classification_report(self.Y_test, y_test_predicted, output_dict=True)
        for col in self.cols:
            val = report[col]
            print(f"{val[self.metrics[0]]:.6f}, {val[self.metrics[1]]:.6f}, {val[self.metrics[2]]:.6f}")

    def svm(self):
        svm = LinearSVC(multi_class="ovr", random_state=0)
        svm.fit(self.tf_X_train, self.Y_train)
        y_test_predicted = svm.predict(self.tf_X_test)

        report = classification_report(self.Y_test, y_test_predicted, output_dict=True)
        for col in self.cols:
            val = report[col]
            print(f"{val[self.metrics[0]]:.6f}, {val[self.metrics[1]]:.6f}, {val[self.metrics[2]]:.6f}")

    def logistic_regression(self):
        lr = LogisticRegression(max_iter=1000, solver='saga')
        lr.fit(self.tf_X_train, self.Y_train)
        y_test_predicted = lr.predict(self.tf_X_test)

        report = classification_report(self.Y_test, y_test_predicted, output_dict=True)
        for col in self.cols:
            val = report[col]
            print(f"{val[self.metrics[0]]:.6f}, {val[self.metrics[1]]:.6f}, {val[self.metrics[2]]:.6f}")

    def multinomial_nb(self):
        nb = MultinomialNB()
        nb.fit(self.tf_X_train, self.Y_train)
        y_test_predicted = nb.predict(self.tf_X_test)

        report = classification_report(self.Y_test, y_test_predicted, output_dict=True)
        for col in self.cols:
            val = report[col]
            print(f"{val[self.metrics[0]]:.6f}, {val[self.metrics[1]]:.6f}, {val[self.metrics[2]]:.6f}")


if __name__ == '__main__':
    AmazonReviewsSentimentAnalysis()

# %%

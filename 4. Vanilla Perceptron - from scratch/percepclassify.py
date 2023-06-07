"""
@course: CSCI 544 Applied NLP
@hw : HW4
@file : Perceptron classification file
@author : Ashwin Chafale
@usc-id : 1990624801
"""

import json
import sys
from collections import defaultdict

from utility import data_preprocessing


def read_test_data(input_file):
    with open(input_file, encoding="UTF-8") as file:
        lines = file.readlines()
        return lines


def read_pickle_file(model_file):
    with open(model_file, mode="r", encoding="UTF-8") as file:
        lines = file.readlines()
        model = [json.loads(line) for line in lines]
        weight = {'True_Fake': model[0], 'Pos_Neg': model[1]}
        return weight


def dump_classification_result(output_dict):
    with open("percepoutput.txt", mode='w', encoding='UTF-8') as output_file:
        for _id in output_dict:
            temp = str(_id) + " " + output_dict[_id][0] + " " + output_dict[_id][1]
            output_file.write(temp + "\n")


class PerceptronClassify:
    def __init__(self, model_weights, data):
        self.weights = model_weights
        self.test_data = data
        self.review_id = []
        self.reviewsList = []

        for reviews in self.test_data:
            review_split = reviews.split(maxsplit=1)
            self.review_id.append(review_split[0])
            self.reviewsList.append(review_split[1])

        self.reviewsList = data_preprocessing(self.reviewsList)
        self.reviewsTokens = [sorted(review.split()) for review in self.reviewsList]

        # generate features
        self.features = self.generate_feature_map()

        # fit the model
        self.model_fit()

    def generate_feature_map(self):
        feature_list = []
        for tokenList in self.reviewsTokens:
            feature_map = defaultdict(int)
            for token in tokenList:
                feature_map[token] = feature_map.get(token, 0) + 1
            feature_list.append(feature_map)
        return feature_list

    def model_fit(self):
        y = {}
        for i in range(len(self.reviewsTokens)):
            _id = self.review_id[i]
            y_TF, y_PN = self.model_classify(self.features[i])
            y[_id] = [y_TF, y_PN]
        dump_classification_result(y)

    def calculate_activation(self, feature, weights):
        activation = 0
        for word in feature:
            if word in weights:
                activation += feature[word] * weights[word]
        return activation + weights['__bias__']

    def model_classify(self, feature):
        activation = self.calculate_activation(feature, self.weights['True_Fake'])
        y_TF = "True" if activation > 0 else "Fake"
        activation = self.calculate_activation(feature, self.weights['Pos_Neg'])
        y_PN = "Pos" if activation > 0 else "Neg"
        return y_TF, y_PN


if __name__ == '__main__':
    path = sys.argv[2]
    test_data = read_test_data(path)

    pathRead = sys.argv[1]
    weights = read_pickle_file(pathRead)

    PerceptronClassify(model_weights=weights, data=test_data)

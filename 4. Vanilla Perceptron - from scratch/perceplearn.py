"""
@course: CSCI 544 Applied NLP
@hw : HW4
@file : Perceptron learn file
@author : Ashwin Chafale
@usc-id : 1990624801
"""

import sys
import numpy as np
import re
import json
from collections import defaultdict
from utility import data_preprocessing


class PerceptronLearn:
    def __init__(self, file_path, iterations=20):
        self.true_fake_label = []
        self.pos_neg_label = []
        self.reviews = []
        # read data from the file
        self.read_file(file_path)
        # pre-process reviews
        self.reviews = data_preprocessing(self.reviews)
        # tokenize the reviews
        self.reviewTokens = [review.split() for review in self.reviews]
        self.reviewTokens = [sorted(tokenList) for tokenList in self.reviewTokens]
        # generate features
        self.features = self.generate_feature_map()
        self.iterations = iterations
        # train the model
        self.perceptron_train()

    def read_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as input_file:
            lines = input_file.readlines()
            for line in lines:
                line_split = line.split(maxsplit=3)
                self.true_fake_label.append(line_split[1])
                self.pos_neg_label.append(line_split[2])
                self.reviews.append(line_split[3])

    def generate_feature_map(self):
        feature_list = []
        for tokenList in self.reviewTokens:
            feature_map = defaultdict(lambda: 1)
            for token in tokenList:
                feature_map[token] = feature_map.get(token, 0) + 1
            feature_list.append(feature_map)
        return feature_list

    def perceptron_train(self):
        weights = {
            "True_Fake": {
                "Vanilla": {"__bias__": 0},
                "Average": {"__bias__": 0}
            },
            "Pos_Neg": {
                "Vanilla": {"__bias__": 0},
                "Average": {"__bias__": 0}
            }
        }
        y = {"True": 1, "Fake": -1, "Pos": 1, "Neg": -1}
        label = {"True_Fake": 1, "Pos_Neg": 1}

        count = 1
        itr = 0
        while itr < self.iterations:
            for i in range(0, len(self.features)):

                label['True_Fake'] = y[self.true_fake_label[i]]
                label['Pos_Neg'] = y[self.pos_neg_label[i]]

                # calculate activation parameter alpha for True-Fake
                alpha = self.calculate_activation(self.features[i], weights['True_Fake'])
                if label['True_Fake'] * alpha <= 0:
                    self.update_weights(self.features[i], weights['True_Fake']['Vanilla'], label['True_Fake'])
                    self.update_weights(self.features[i], weights['True_Fake']['Average'], label['True_Fake'], count)

                # calculate activation parameter alpha for Pos-Neg
                alpha = self.calculate_activation(self.features[i], weights['Pos_Neg'])
                if label['Pos_Neg'] * alpha <= 0:
                    self.update_weights(self.features[i], weights['Pos_Neg']['Vanilla'], label['Pos_Neg'])
                    self.update_weights(self.features[i], weights['Pos_Neg']['Average'], label['Pos_Neg'], count)

                count += 1
            itr += 1

        self.calculate_final_avg_weights(weights['True_Fake']['Vanilla'], weights['True_Fake']['Average'], count)
        self.calculate_final_avg_weights(weights['Pos_Neg']['Vanilla'], weights['Pos_Neg']['Average'], count)

        self.pickle_model(weights)

    def calculate_activation(self, feature, weights):
        activation = 0
        for word in feature:
            if word in weights['Vanilla']:
                activation += feature[word] * weights['Vanilla'][word]
            else:
                weights['Vanilla'][word] = 0
                weights['Average'][word] = 0
                activation += 0
        return activation + weights['Vanilla']['__bias__']

    def update_weights(self, feature, weights, label, count=1):
        # update the weights
        for token in feature:
            weights[token] += label * feature[token] * count
        # update the bias
        weights['__bias__'] += label * count

    def calculate_final_avg_weights(self, vanilla_weights, avg_weights, count):
        for word in avg_weights:
            avg_weights[word] = vanilla_weights[word] - (avg_weights[word] / count)

    def pickle_model(self, weights):
        print("Writing Vanilla perceptron pickle file . . .")
        with open("vanillamodel.txt", mode="w", encoding="utf-8") as vanilla_output_file:
            vanilla_output_file.write(json.dumps(weights['True_Fake']['Vanilla']))
            vanilla_output_file.write("\n")
            vanilla_output_file.write(json.dumps(weights['Pos_Neg']['Vanilla']))
            vanilla_output_file.write("\n")

        print("Writing Average perceptron pickle file . . .")
        with open("averagedmodel.txt", mode="w", encoding="utf-8") as average_output_file:
            average_output_file.write(json.dumps(weights['True_Fake']['Average']))
            average_output_file.write("\n")
            average_output_file.write(json.dumps(weights['Pos_Neg']['Average']))
            average_output_file.write("\n")


if __name__ == '__main__':
    filePath = sys.argv[1]
    # filePath = "./perceptron-training-data/train-labeled.txt"
    PerceptronLearn(filePath, iterations=20)

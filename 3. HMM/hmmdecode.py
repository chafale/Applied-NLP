"""
@course: CSCI 544 Applied NLP
@hw : HW3
@file : HMM Decode file
@author : Ashwin Chafale
@usc-id : 1990624801
"""

import math
import sys
from collections import defaultdict


class HmmDecode:
    def __init__(self, model_path):
        with open(model_path, 'r', encoding='utf-8') as model_file:
            self.model = model_file.readlines()
            self.words_set = eval(self.model[0])
            self.tags_dict = eval(self.model[1])
            self.tags_set = self.tags_dict.keys() - {"|<=START=>|", "|<=END=>|"}
            self.transition_dict = eval(self.model[2])
            self.emission_dict = eval(self.model[3])

        # initial opening tags
        self.opening_tags = self.get_opening_tags(top_k=5)

        # result
        self.predictions_result = []

    def decode(self, file_path):
        """
        Viterbi Algorithm
        :param file_path: file path to decode the text
        :return: None
        """
        with open(file_path, 'r', encoding='utf-8') as input_file:
            lines = input_file.readlines()
            for line in lines:
                words = line.split()

                dp_matrix = {}
                base_tags = {}

                # initialize probability_matrix and base_tag dict
                for idx, word in enumerate(words):
                    word_tuple = (word, idx)
                    dp_matrix[word_tuple] = {}
                    base_tags[word_tuple] = {}
                    for tag in self.tags_dict:
                        if tag == "|<=START=>|" or tag == "|<=END=>|":
                            continue
                        else:
                            dp_matrix[word_tuple][tag] = -math.inf
                            base_tags[word_tuple][tag] = ""

                # initialize the first column of probability_matrix
                first_word = (words[0], 0)
                for tag in self.tags_set:
                    transition_tuple = ("|<=START=>|", tag)
                    emission_tuple = (tag, first_word[0])
                    if first_word[0] in self.words_set:
                        if emission_tuple in self.emission_dict:
                            dp_matrix[first_word][tag] = math.log(self.transition_dict[transition_tuple]) + \
                                                         math.log(self.emission_dict[emission_tuple])
                            base_tags[first_word][tag] = "|<=START=>|"
                    else:
                        dp_matrix[first_word][tag] = math.log(self.transition_dict[transition_tuple])
                        base_tags[first_word][tag] = "|<=START=>|"

                # remaining columns
                for idx, word in enumerate(words[1:]):
                    idx += 1
                    word_tuple = (word, idx)
                    for tag in self.tags_set:
                        # seen word
                        if word_tuple[0] in self.words_set:
                            emission_tuple = (tag, word_tuple[0])
                            if emission_tuple in self.emission_dict:
                                for prev_tag in self.tags_set:
                                    if dp_matrix[(words[idx - 1], idx - 1)][prev_tag] != -math.inf:
                                        transition_tuple = (prev_tag, tag)
                                        probability = dp_matrix[(words[idx - 1], idx - 1)][prev_tag] + \
                                                      math.log(self.transition_dict[transition_tuple]) + \
                                                      math.log(self.emission_dict[emission_tuple])
                                        if probability > dp_matrix[word_tuple][tag]:
                                            dp_matrix[word_tuple][tag] = probability
                                            base_tags[word_tuple][tag] = prev_tag
                        # unseen word
                        else:
                            for prev_tag in self.tags_set:
                                if dp_matrix[(words[idx - 1], idx - 1)][prev_tag] != -math.inf:
                                    transition_tuple = (prev_tag, tag)
                                    probability = dp_matrix[(words[idx - 1], idx - 1)][prev_tag] + \
                                                  math.log(self.transition_dict[transition_tuple])
                                    if probability > dp_matrix[word_tuple][tag]:
                                        dp_matrix[word_tuple][tag] = probability
                                        base_tags[word_tuple][tag] = prev_tag

                # last column
                best_probability = -math.inf
                best_tag = ""
                last_word = (words[-1], len(words) - 1)
                for tag in self.tags_set:
                    if dp_matrix[last_word][tag] != -math.inf:
                        transition_tuple = (tag, "|<=END=>|")
                        probability = dp_matrix[last_word][tag] + math.log(self.transition_dict[transition_tuple])
                        if probability > best_probability:
                            best_probability = probability
                            best_tag = tag

                # backtracking
                tags = []
                curr_word = last_word[0]
                curr_idx = len(words) - 1
                curr_tag = best_tag
                while curr_tag != "|<=START=>|":
                    tags.append(curr_tag)
                    curr_tag = base_tags[(curr_word, curr_idx)][curr_tag]
                    curr_word = words[curr_idx - 1]
                    curr_idx -= 1

                tags.reverse()
                prediction = ''
                for idx, word in enumerate(words):
                    prediction += '{}/{} '.format(word, tags[idx])

                self.predictions_result.append(prediction.strip())

    def get_opening_tags(self, top_k):
        """
        function to get the opening tags for initial sequence
        :param top_k: int
        :return: dict(string, int)
        """
        open_tags = defaultdict(set)
        for tag, word in self.emission_dict:
            open_tags[tag].add(word)
        open_tags = {tag: len(open_tags[tag]) for tag in open_tags}
        open_tags = sorted(open_tags.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return open_tags

    def output_result(self):
        with open('hmmoutput.txt', 'w', encoding='utf-8') as output_file:
            output_file.truncate(0)
            for prediction in self.predictions_result:
                output_file.write(f'{prediction}\n')


if __name__ == '__main__':
    print("======== HMM Decode ========")
    testFile = sys.argv[1]
    modelFile = "hmmmodel.txt"
    hmm_decode = HmmDecode(modelFile)
    hmm_decode.decode(testFile)
    print("Decide completed. Writing to hmmoutput.txt file ...")
    hmm_decode.output_result()
    print("Writing completed.")

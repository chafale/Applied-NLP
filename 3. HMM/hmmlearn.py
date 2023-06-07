"""
@course: CSCI 544 Applied NLP
@hw : HW3
@file : HMM Learn file
@author : Ashwin Chafale
@usc-id : 1990624801
"""

import sys


class HmmLearn:
    """
    HMM Learn model class to build emission and transition matrix
    """

    def __init__(self):
        self.unique_words = set()
        self.unique_tags_dict = {"|<=START=>|": 0, "|<=END=>|": 0}
        self.tags_counter = {}
        self.emission_dict = {}
        self.transition_dict = {}

    def build_model(self, file_path):
        """
        function to learn emission and transition probabilities from the learning data
        :param file_path: file_path to the tagged data file
        :return: None
        """
        with open(file_path, encoding='utf-8') as input_file:
            lines = input_file.readlines()
            for each_line in lines:
                tokens = each_line.split()
                previous_tag = "|<=START=>|"

                for each_token in tokens:
                    word, tag = each_token.rsplit("/", 1)
                    # word = word.lower()

                    # add word to unique_words set
                    self.unique_words.add(word)

                    # add tag to unique_tags_dict
                    self.unique_tags_dict[tag] = self.unique_tags_dict.get(tag, 0) + 1

                    # initialize emission probabilities
                    emission_tuple = (tag, word)
                    self.emission_dict[emission_tuple] = self.emission_dict.get(emission_tuple, 0) + 1

                    # initialize transition probabilities
                    transition_tuple = (previous_tag, tag)
                    self.transition_dict[transition_tuple] = self.transition_dict.get(transition_tuple, 0) + 1

                    previous_tag = tag

                    # adding END tag to transition dict
                    self.transition_dict[(tag, "|<=END=>|")] = self.transition_dict.get((tag, "|<=END=>|"), 0) + 1

                    self.unique_tags_dict["|<=START=>|"] += 1
                    self.unique_tags_dict["|<=END=>|"] += 1

        # perform smoothing
        self.smoothing()

        # calculate transition probability
        for transition in self.transition_dict:
            self.transition_dict[transition] = self.transition_dict[transition] / self.tags_counter[transition[0]]

        # calculate emission probability
        for emission in self.emission_dict:
            self.emission_dict[emission] = self.emission_dict[emission] / self.tags_counter[emission[0]]

    def smoothing(self):
        for tag in self.unique_tags_dict:
            self.tags_counter[tag] = self.unique_tags_dict[tag]

        # smoothing for transition matrix
        for tag_1 in self.unique_tags_dict:
            for tag_2 in self.unique_tags_dict:
                if tag_1 == "|<=START=>|" and tag_2 == "|<=END=>|":
                    continue
                if tag_1 == "|<=END=>|" or tag_2 == "|<=START=>|":
                    continue

                if (tag_1, tag_2) not in self.transition_dict:
                    self.transition_dict[(tag_1, tag_2)] = 1
                    self.tags_counter[tag_1] += 1

    def save_model(self, file_name):
        with open(file_name, "w", encoding="utf-8") as output_file:
            output_file.truncate(0)
            output_file.write(f'{self.unique_words}\n')
            output_file.write(f'{self.unique_tags_dict}\n')
            output_file.write(f'{self.transition_dict}\n')
            output_file.write(f'{self.emission_dict}\n')


if __name__ == '__main__':
    print("======== HMM Learning ========")
    filePath = sys.argv[1]
    # filePath = "./hmm-training-data/it_isdt_train_tagged.txt"
    hmm = HmmLearn()
    hmm.build_model(filePath)
    print("Learning completed. Writing to hmmmodel.txt file ...")
    hmm.save_model("hmmmodel.txt")
    print("Writing completed.")

import json
import csv
import string

"""
Class description: For every word in the file of phrases, create a csv file that prints all words on the left column,
    and then any number of potential categories that can differentiate that word from other words
"""


class WordCategorizer:
    def __init__(self, word_file='text_files/florida_beach_GIRLS_new', word_list=None):
        # list of words in the file
        self.categories = []
        self.highest_index = 0
        self.data_table = {}
        self.word_file = open(word_file, 'r')
        self.phrases = word_list if word_list is not None else self.word_file.readlines()
        self.number_of_words = 0
        self.create_all_category_headers()
        # self.csv_file = csv.

    # Use the category list to update all category headings
    def create_all_category_headers(self):
        dummy_word = WordUtilities()
        for category in dummy_word.categories:
            self.categories.append(category)

    # start by filling the categories dictionary with the list of words and their frequencies
    def create_word_category_list(self, phrases):
        for phrase in phrases:
            # for each line
            for word in phrase.split(" "):
                word = word.replace("\n", "")

                # add all words to category list
                if word != "":
                    self.update_data_table(word)

    def update_data_table(self, word):
        # add new word if not in table
        if word not in self.data_table:
            self.add_new_word(word)
            return

        self.data_table[word].update_frequency()

    # add all words to category list
    def add_new_word(self, word):
        # add new word if not in table
        self.number_of_words += 1
        self.data_table[word] = WordUtilities(name=word)
        self.data_table[word].categories["ID"] = self.number_of_words

    # print all rows in data table
    def print_data_table(self):
        for word in self.data_table:
            print(f"{word}: {self.data_table[word].categories}")

    # get the unique numerical id map for all words
    def create_ID_map(self):
        ID_map = {}
        for word in self.data_table:
            ID_map[word] = self.data_table[word].categories["ID"]
        return ID_map

    def as_csv(self):
        file = open('word_info.csv', 'w', newline='')

        with file:
            # identifying header
            header = self.categories
            writer = csv.DictWriter(file, fieldnames=header)

            # writing data row-wise into the csv file
            writer.writeheader()
            for word in self.data_table:
                writer.writerow(self.data_table[word].categories)


# Keep track of and update the dictionary of categories
class WordUtilities:
    def __init__(self, name=" "):
        # list of words in the file
        self.words = []
        self.highest_index = 0
        self.data_table = {}
        self.number_of_words = 0
        self.name = name
        self.initialize_categories()
        self.ascii_arr = [ord(letter)-96 for letter in self.name]
        # self.letter_frequencies = self.most_frequent_positions()
        while len(self.ascii_arr) < 20:
            self.ascii_arr.append(0)

    def create_add_ons(self):
        self.add_ons = []

        # one-hot encode first and last labels
        for idx in range(20):
            if idx >= len(self.name):
                self.add_ons.extend([0 for _ in range(26)])
            else:
                letter_arr = [0 for _ in range(26)]
                if self.name[idx] in string.ascii_lowercase:
                    letter_arr[ord(self.name[idx])-97] = 1
                self.add_ons.extend(letter_arr)

        # first_letter_arr = [0 for _ in range(26)]
        # if self.name[0] in string.ascii_lowercase:
        #     first_letter_arr[ord(self.name[0])-97] = 1
        #
        # middle_letter_arr = [0 for _ in range(26)]
        # if self.name[int(len(self.name)/2)] in string.ascii_lowercase:
        #     middle_letter_arr[ord(self.name[int(len(self.name)/2)]) - 97] = 1
        #
        # last_letter_arr = [0 for _ in range(26)]
        # if self.name[-1] in string.ascii_lowercase:
        #     last_letter_arr[ord(self.name[-1])-97] = 1
        #
        # self.add_ons.extend(first_letter_arr)
        # self.add_ons.extend(middle_letter_arr)
        # self.add_ons.extend(last_letter_arr)

        self.add_ons.extend(self.most_frequent_positions())

        return self.add_ons

    #def first_and_last_letter_one_hot_encoded(self):


    def initialize_categories(self):
        self.categories = {
            "name": self.name,
            "ID": 0,
            "frequency": 1,
            # "first_letter": ord(self.name[0])-96,
            # "last_letter": ord(self.name[-1])-96,
            "most_frequent_position": 0,
            "length": len(self.name),
            "number_of_syllables": self.count_syllables()}

    # update word's frequency
    def update_frequency(self):
        self.categories["frequency"] += 1

    def most_frequent_positions(self):
        letters = set(string.ascii_lowercase)
        letters.update(string.digits)
        letter_frequencies = {}

        for letter in letters:
            if letter == '_':
                continue
            letter_frequencies[letter] = 0

        for letter in self.name:
            if letter == '_':
                continue
            letter_frequencies[letter] += 1

        return list(letter_frequencies.values())

    # Count number of syllables by counting the vowels in each word plus all words
    # that end with "le". Subtract 1 from the syllable count if the word
    # ends with 'es', 'ed', or 'e' (not 'le')
    def count_syllables(self):
        word = self.name
        words_in_speech = str(self.name).split(" ")
        self.syllables_count = 0
        vowels = ["a", "e", "i", "o", "u"]


        # first count all vowels in each word
        for letter in word:
            if letter in vowels:
                self.syllables_count += 1

        # count all words that end with 'le'
        if word.endswith("le"):
            self.syllables_count += 1

        # subtract 1 form the syllable count if the word ends with 'es', 'ed',
        # or 'e' (not 'le')
        if word.endswith('es') or word.endswith('ed') \
                or (word.endswith('e') and not word.endswith('le')):
            self.syllables_count -= 1

        return self.syllables_count


if __name__ == "__main__":
    w = WordUtilities("sarasota")
    print(w.most_frequent_positions())

    # file = 'text_files/florida_beach_GIRLS_old'
    # w = WordCategorizer(file)
    # # print(w.phrases)
    # w.create_word_category_list(w.phrases)
    # w.print_data_table()
    # w.as_csv()
    # print(w.create_ID_map())

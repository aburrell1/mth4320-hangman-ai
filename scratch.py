import random
import string

from hangman_engine import HangmanEngine
from create_csv_categories import WordUtilities
import csv


# extra feature lists to try
def letter_list(word):
    new_l = []
    lowercase_letters = string.ascii_lowercase
    for letter in lowercase_letters:
        if letter in word:
            new_l.append(1)
        else:
            new_l.append(0)

    return new_l


if __name__ == '__main__':

    phrase_list = open('text_files/countries').readlines()
    f_large = open('practice_words')
    new_f = open('text_files/countries_formatted', 'w')
    s = set()
    s.update(f_large.readlines())
    print(s)

    for word in phrase_list:
        new_f.write(word.lower())

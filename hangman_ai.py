import csv
import string
import random
import xgboost as xgb
import numpy as np
import tensorflow as tf
import tensorflow.python.framework.ops
from keras import Input, Model
from keras.layers import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LeakyReLU, Bidirectional, LSTM, SimpleRNN
from tensorflow.keras.utils import to_categorical
from tensorflow.python.layers.core import Dropout
from create_csv_categories import *
from project1.hangman_engine import HangmanEngine
from sklearn.utils import class_weight
import scratch


class HangmanAI:
    def __init__(self, training_file='text_files/common_phrases'):
        self.training_file = training_file
        self.new_words = WordCategorizer()
        self.letter_frequencies = {}
        self.seen_words = set()
        self.model = Sequential()
        self.isWrongOnce = False
        # self.old_words = WordCategorizer(word_file='text_files/florida_beach_GIRLS_old')

    def update_most_common_letters(self, word):
        if self.letter_frequencies == {}:
            for letter in string.ascii_lowercase:
                self.letter_frequencies[letter] = 0

        for letter in word:
            self.letter_frequencies[letter] += 1

        self.letter_frequencies = dict(reversed(sorted(self.letter_frequencies.items(), key=lambda item: item[1])))

    def build_model(self, practice_data_arr):
        print(self.letter_frequencies)
        number_of_used_columns = len(practice_data_arr[0]) - 1
        # X_train = X_train.reshape(X_train.shape[0], 26 * 20).astype('float32')
        # X_test = X_test.reshape(X_test.shape[0], 26 * 20).astype('float32')
        #
        # self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(number_of_used_columns, 1)))
        # self.model.add(MaxPooling1D(pool_size=3))
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.3))
        # self.model.add(Flatten())

        # self.model.add(Dense(128, input_shape=(number_of_used_columns,), activation='relu',
        #                 kernel_regularizer=l1_l2(l1=0.0, l2=0.001)))  # 2 input features
        # self.model.add(Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.0, l2=0.0001)))
        # self.model.add(Dense(len(np.unique(labels)), activation='softmax'))  # Number of classes as output neurons
        self.model.add(Dense(400, input_shape=(number_of_used_columns,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(400, activation='relu', input_shape=(number_of_used_columns,)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(400, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(len(np.unique(labels)), activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # print(self.letter_frequencies)

    def fit(self, X_train, y_train, X_val, y_val):
        callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=15)
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=32, callbacks=[callback])

    def xgb_build_and_fit(self, X_train, y_train, X_val, y_val):
        self.model = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=10)
        self.model.fit(X_train, y_train)
        result_train = self.model.score(X_train, y_train)
        print("Accuracy : {}".format(result_train))

    # decides whether to guess whole phrase using deep learning or just a letter at a time
    # with the letter frequency table
    def decide_guess_mode(self, blank_phrase):
        return self.best_prediction_value >= 0.6 and self.get_blank_amount(blank_phrase) <= 0.5

    def update_word_categorizer(self, word):
        self.new_words.update_data_table(word)

    def process_phrase(self, phrase):
        phrase = phrase.split(" ")
        phrase_decoded = []
        for word in phrase:
            phrase_decoded.append(WordUtilities(word).categories)
            self.new_words.update_data_table(word)
        # category_list = self.new_words.create_word_category_list(phrase)

        return phrase_decoded

    def create_training_data(self):
        f = open('text_files/practice_words', 'w')

        # create list of individual words
        old_file = open(self.training_file)
        all_lines = old_file.readlines()
        all_lines.remove(all_lines[-1])

        # ext
        for line in all_lines:
            for word in line.split(" "):
                word = word.replace("\n", "")
                if word == "":
                    continue

                # add word to dictionaries
                self.update_most_common_letters(word)
                self.seen_words.add(word)
                self.new_words.update_data_table(word)
                f.write(f'{word}\n')

        word_list = open('text_files/practice_words').readlines()
        hangman_engine = HangmanEngine('practice_words')

        # desired_length = 30000

        u = [word.replace("\n", "") for word in word_list]
        u = u * 300

        for word in self.new_words.data_table:
            # if self.new_words.data_table[word].categories['frequency'] < 400:
            for _ in range(1000):
                u.append(word.replace("\n", ""))

        # shuffle the dataset with the same seed
        random.Random(4).shuffle(u)
        # print(len(u))

        blank_intensies = []
        # create a list of blank intensities ranging from 0 to 0.8
        for i in range(len(u)):
            frac = (i+1)/(len(u)+1) if (i+1)/(len(u)+1) < 0.15 else (i+1)/(len(u)+1) * 0.6
            blank_intensies.append(frac)
        random.Random(4).shuffle(blank_intensies)
        b = []
        for i, phrase in enumerate(u):
            b.append(hangman_engine.blank_phrases(blank_intensity=blank_intensies[i], phrases_list=[phrase])[0])

        # b = hangman_engine.blank_phrases(blank_intensity=random.randrange(10, 70) / 100, phrases_list=u)
        b = [word.replace("\n", "") for word in b]
        b.extend([word for word in word_list])

        ans_file = open('practice_answers.csv', 'w', newline='')
        ans_dict_writer = csv.writer(ans_file)

        for u_word, b_word in zip(u, b):
            b_word_utilities = WordUtilities(b_word)
            answer_word_encoded = []
            answer_word_encoded.extend(list(b_word_utilities.categories.values())[2:])

            add_ons = b_word_utilities.create_add_ons()
            if add_ons is not []:
                answer_word_encoded.extend(add_ons)

            row = [u_word]
            row.extend(answer_word_encoded)
            ans_dict_writer.writerow(row)

    def make_prediction(self, b_word):
        # encode the word into array
        encoded_word = np.array(encode_word(b_word))

        # make numerical prediction
        self.prediction_arr = hangman_ai.model.predict(encoded_word)
        self.best_prediction_value = max(self.prediction_arr[0])

        # get top 10 numerical predictions
        self.top_10_predictions = np.argpartition(self.prediction_arr, -100)[0][-100:][::-1]
        self.top_10_predictions = self.top_10_predictions[np.argsort(-self.prediction_arr[0][self.top_10_predictions])]

        # get words representing top 10 predictions
        self.probability_classes = [encoder.inverse_transform([element])[0] for element in self.top_10_predictions]

        # choose the word through this drawn out process if we are trying to guess entire word
        self.word_chosen = self.probability_classes[0]

        return self.word_chosen

    def get_blank_amount(self, blanked_word):
        blank_count = 0
        for char in blanked_word:
            if char == '_':
                blank_count += 1

        return blank_count / len(blanked_word)

    # get top 10 predictions for the letter based on letter frequency
    def top_10_letter_predictions(self, blanked_word):
        print(self.letter_frequencies.keys())
        # self.letter_frequencies = dict(reversed(sorted(self.letter_frequencies.items(), key=lambda item: item[1])))

        predictions = set()
        predictions.update(list(self.letter_frequencies.keys()))


        return list(predictions)

    # if we have one letter left, then use this method to try and get it
    def predict_word_with_one_blank(self, blanked_word):
        predicted_words = []
        for letter in self.letter_frequencies:
            predicted_word = list(blanked_word[:])
            blank_idx = predicted_word.index("_")
            predicted_word[blank_idx] = letter

            if predicted_word in self.seen_words:
                predicted_words.append(predicted_word)

        return predicted_words


def encode_word(word, u_word=None):
    row = []
    row.extend(list(WordUtilities(word).categories.values())[2:])

    add_ons = WordUtilities(word).create_add_ons()
    if add_ons:
        row.extend(add_ons)
    # new content (remove if not working)
    # if u_word is not None:
    #     row[0] = np.log(hangman_ai.new_words.data_table[u_word].categories['frequency'])
    # row.extend(scratch.letter_list(word))
    return np.array([row])


if __name__ == "__main__":
    vocab_file = 'text_files/countries_formatted'
    hangman_ai = HangmanAI(vocab_file)
    hangman_ai.create_training_data()

    practice_guess_arr = []
    with open('practice_answers.csv', 'r') as f:
        practice_guess = csv.reader(f)

        for row in practice_guess:
            practice_guess_arr.append(row)

    labels = np.array([row[0] for row in practice_guess_arr])

    # new content (remove if not working)
    start_of_categories_idx = 1
    data = [row[start_of_categories_idx:] for row in practice_guess_arr]

    data = np.array(data, dtype=float)

    # Convert string labels to integers
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print(encoded_labels)

    # One hot encode the labels
    one_hot_labels = to_categorical(encoded_labels)
    print(one_hot_labels)

    # Split the data into training and testing sets

    # X_train, X_test, y_train, y_test = prepare_testing_data(vocab_file)
    X_train, X_test, y_train, y_test = train_test_split(data, one_hot_labels, test_size=0.2, random_state=42)

    # Define a simple neural network model
    hangman_ai.build_model(practice_guess_arr)
    hangman_ai.fit(X_train, y_train, X_test, y_test)

    hangman_game = HangmanEngine(vocab_file, 2)
    hangman_game.play_ai_game(hangman_ai, deck_size=25, blank_intensity=1, number_of_tries=8)





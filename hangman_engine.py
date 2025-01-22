import random
import string

# This class runs the game of hangman


class HangmanEngine:
    def __init__(self, file_of_phrases, max_phrase_length=20):
        self.file_of_phrases = file_of_phrases
        self.phrases = []
        self.min_phrase_length = 1
        self.max_phrase_length = max_phrase_length
        self.accuracy = 0
        self.score = 0
        self.blank_phrase_list = []
        self.word_IDs = {}
        self.ID_words = {}
        self.format_phrases()

    """
    Description: Format all the phrases used to remove numbers, parentheses, etc.
    """
    def format_phrases(self):
        f = open(self.file_of_phrases)

        # only lowercase and uppercase letters wanted in our phrase list
        acceptable_characters = set()
        lowercase = list(string.ascii_lowercase)
        uppercase = list(string.ascii_uppercase)
        acceptable_characters.update(lowercase + uppercase + list(' '))

        # Create a new phrase with only letters and spaces
        for phrase in f.readlines():
            phrase_too_long = len(phrase.split(' ')) > self.max_phrase_length
            if phrase_too_long:
                print(phrase.split(" "))
                continue

            new_phrase = []
            # append all letters in the phrase to the new_phrase
            for c in phrase:
                if c in acceptable_characters:
                    # only append one space and only if the phrase is less than the max phrase length
                    contains_extra_spaces = (c == ' ' and len(new_phrase) > 0 and new_phrase[-1] == ' ')
                    if not contains_extra_spaces:
                        new_phrase.append(c.lower())

            # Add the formatted phrase to the list of phrases
            self.phrases.append(''.join(new_phrase))

    """
    Description: For every phrase in the file, set a certain number of characters blank.
    Parameters: blank_intensity --> how much of the phrase we want blanked
    """
    def blank_phrases(self, blank_intensity, phrases_list=None):
        phrases = self.phrases if phrases_list is None else phrases_list
        blank_phrase_list = []
        for phrase in phrases:
            # make a certain percentage of the characters in the phrase blank at random points
            tmp = set()
            while len(tmp) <= int(blank_intensity * len(phrase)):
                tmp.add(random.randint(0, len(phrase)))

            # create a new blank phrase and append it to the list of blank phrases
            blank_indexes = set(tmp)
            blank_phrase = list(phrase[:])
            for idx in range(len(blank_phrase)):
                if idx in blank_indexes and blank_phrase[idx] != ' ':
                    blank_phrase[idx] = '_'

            blank_phrase_list.append(''.join(blank_phrase))

        return blank_phrase_list

    """
    Description: Create deck of phrases for a single game.
    :parameter deck_size --> the size of the deck
    :parameter max_phrase_length --> the maximum phrase length
    :parameter blank_intensity --> the number of characters in the phrase that will be blanked
    """
    def create_deck(self, deck_size, blank_intensity):
        blank_phrases = self.blank_phrases(blank_intensity)

        # create current deck from random elements in the blank phrase list. This is a set to prevent adding duplicates
        blank_deck = []
        unblank_deck = []
        tmp = set()
        # create list of random indices to grab from for potential deck phrases
        while len(tmp) < deck_size:
            tmp.add(random.randint(0, len(blank_phrases)))

        # fill up the current_deck with the number of desired phrases
        for idx in range(len(blank_phrases)):
            # select phrases less than the max phrase length (length is determined by number of words in the phrase
            current_blank_phrase = blank_phrases[idx]
            isWithinPhraseSizeLimits = len(str(current_blank_phrase).split(" ")) < self.max_phrase_length
            if isWithinPhraseSizeLimits and idx in tmp:
                blank_deck.append(current_blank_phrase)
                unblank_deck.append(self.phrases[idx])

        # self.create_answers_csv(unblank_deck)
        return unblank_deck, blank_deck

    """
    Description: Create deck of phrases for a single game.
    :parameter 
    """
    def play_game(self, deck_size, blank_intensity, number_of_tries=10):
        unblank_deck, blank_deck = self.create_deck(deck_size, blank_intensity)

        # for all phrases in the deck
        idx = 0
        while idx < len(blank_deck):
            tries_left = number_of_tries
            current_phrase = unblank_deck[idx]
            current_blank_phrase = blank_deck[idx]
            print(current_blank_phrase)

            # how many points are accumulated depends on how many tries it took to guess correctly
            score_for_current_guess = 1

            # for every phrase, try to guess phrase within n number of tries
            while tries_left > 0:
                # Choose what kind of guess to make
                guess_phrase = input("Guess letter (l) or entire phrase (p)?")
                if guess_phrase == "l":
                    self.guess_entire_phrase = False
                else:
                    self.guess_entire_phrase = True

                # Follow this path if trying to guess entire phrase
                if self.guess_entire_phrase:
                    ########################################################################
                    # change this variable when using the AI to guess
                    guess = input("Guess whole phrase")
                    ########################################################################

                    if guess.lower() == current_phrase.lower():
                        print("Correctly guessed phrase")
                        self.accuracy += (1 / deck_size)
                        self.score += score_for_current_guess * 100
                        print(f"Current Score: {int(self.score)}")
                        break
                    else:
                        print("Wrong. Try again.")
                        score_for_current_guess -= (1 / number_of_tries)
                        tries_left -= 1

                # guess only a letter in the phrase
                else:
                    # Change this when using the AI to guess
                    guess = input("Guess a single letter:")

                    if guess.lower() in current_phrase.lower():
                        current_blank_phrase = list(current_blank_phrase[:])
                        for i in range(len(current_blank_phrase)):
                            if current_phrase[i] == guess:
                                current_blank_phrase[i] = guess

                        current_blank_phrase = ''.join(current_blank_phrase)

                        # entire phrase found
                        if "_" not in current_blank_phrase:
                            print("Letter found in phrase")
                            print(f"Updated Phrase: {current_blank_phrase}")
                            print("Correctly guessed phrase")
                            self.accuracy += (1 / deck_size)
                            self.score += score_for_current_guess * 100
                            print(f"Current Score: {int(self.score)}")
                            break

                        # more blanks left
                        else:
                            print("Letter found in phrase")
                            print(f"Updated Phrase: {current_blank_phrase}")
                            # self.accuracy += (0.5 / deck_size)
                            score_for_current_guess -= (0.1 / number_of_tries)
                            # self.score += score_for_current_guess * 100
                            # print(f"Current Score: {int(self.score)}")

                    else:
                        print("Letter not in phrase. Try again.")
                        score_for_current_guess -= (0.5 / number_of_tries)
                        tries_left -= 1

            if tries_left == 0:
                print(f"Did not guess phrase. Moving to next phrase. This was the correct phrase: {current_phrase}")

            idx += 1

        print(f"Score: {int(self.score)}")
        print(f"Accuracy: {self.accuracy}")
        return self.accuracy

    def play_ai_game(self, hangman_ai, deck_size, blank_intensity, number_of_tries=3):
        # hangman_ai = HangmanAI()
        unblank_deck, blank_deck = self.create_deck(deck_size, blank_intensity)

        # for all phrases in the deck
        idx = 0
        while idx < len(blank_deck):
            tries_left = number_of_tries
            current_phrase = unblank_deck[idx]
            current_blank_phrase = blank_deck[idx]
            print(current_blank_phrase)

            # the letters in the current word for the sake of guessing letters
            letter_arr = list(hangman_ai.letter_frequencies.keys())

            # how many points are accumulated depends on how many tries it took to guess correctly
            score_for_current_guess = 1

            # for every phrase, try to guess phrase within n number of tries
            while tries_left > 0:
                # Choose what kind of guess to make
                # guess_phrase = input("Guess letter (l) or entire phrase (p)?")
                # if we are confident in our prediction (more than 0.5), then guess whole phrase
                self.word_guess = hangman_ai.make_prediction(current_blank_phrase) if not hangman_ai.isWrongOnce else None
                self.guess_entire_phrase = (hangman_ai.best_prediction_value >= 0.6 and hangman_ai.get_blank_amount(current_blank_phrase) <= 0.5)

                # Follow this path if trying to guess entire phrase
                if self.guess_entire_phrase:
                    ########################################################################
                    # change this variable when using the AI to guess
                    # guess = input("Guess whole phrase")
                    guess = hangman_ai.word_chosen
                    ########################################################################

                    if guess.lower() == current_phrase.lower():
                        print(f"Correctly guessed phrase: {current_phrase}")
                        hangman_ai.isWrongOnce = False
                        self.accuracy += (1 / deck_size)
                        self.score += score_for_current_guess
                        print(f"Current Score: {self.score}")
                        break

                    else:
                        print("Wrong. Try again.")
                        hangman_ai.isWrongOnce = True
                        hangman_ai.probability_classes.pop(0)

                        while len(hangman_ai.probability_classes[0]) != len(current_blank_phrase):
                            hangman_ai.probability_classes.pop(0)

                        hangman_ai.word_chosen = hangman_ai.probability_classes[0]

                        print(f"Probability score for guess: {hangman_ai.best_prediction_value}, guess: {guess}")
                        score_for_current_guess -= (1 / number_of_tries)
                        tries_left -= 1

                # guess only a letter in the phrase
                else:
                    # print(hangman_ai.letter_frequencies)
                    ##########################################################
                    # Change this when using the AI to guess
                    # guess = input("Guess a single letter:")
                    guess = letter_arr[0]
                    ##########################################################

                    # if we found a letter in the phrase
                    if guess.lower() in current_phrase.lower():
                        current_blank_phrase = list(current_blank_phrase[:])
                        for i in range(len(current_blank_phrase)):
                            if current_phrase[i] == guess:
                                current_blank_phrase[i] = guess

                        current_blank_phrase = ''.join(current_blank_phrase)

                        # entire phrase found
                        if "_" not in current_blank_phrase:
                            print("Letter found in phrase")
                            print(f"Updated Phrase: {current_blank_phrase}")
                            print("Correctly guessed phrase")
                            self.accuracy += (1 / deck_size)
                            self.score += score_for_current_guess
                            print(f"Current Score: {int(self.score)}")
                            break

                        # more blanks left
                        else:
                            print("Letter found in phrase")
                            print(f"Updated Phrase: {current_blank_phrase}")
                            # self.accuracy += (0.5 / deck_size)
                            score_for_current_guess -= (0.2 / number_of_tries)
                            letter_arr.pop(0)
                            # letter_idx += 1
                            # letter_arr.pop(letter_idx)
                            # self.score += score_for_current_guess * 100
                            # print(f"Current Score: {int(self.score)}")

                    else:
                        # pop the letter if not in the phrase
                        letter_arr.pop(0)
                        # letter_idx += 1

                        print(f"Letter guess: {letter_arr[0]}")
                        print("Letter not in phrase. Try again.")
                        score_for_current_guess -= (0.5 / number_of_tries)

                        tries_left -= 1

            if tries_left == 0:
                print(f"Did not guess phrase. Moving to next phrase. This was the correct phrase: {current_phrase}")

            idx += 1

        print(f"Score: {self.score} / {deck_size}")
        print(f"Accuracy: {self.accuracy}")
        return self.accuracy


if __name__ == "__main__":
    file = 'text_files/countries_formatted'
    h = HangmanEngine(file, 10)
    #print(h.play_ai_game())
    print(h.play_game(3, 1))

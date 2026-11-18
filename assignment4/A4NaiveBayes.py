# ---------------------------------------------------
# Irina Khan
# SYDE 572 Assignment 4
#
# Objective: 
# This assignment is aimed to apply the Naive Bayes 
# classificaiton method to help sort customer reviews 
# into 'positive' reviews or 'negative' reviews. In 
# the dataset 'all_sentiment_shuffled.txt', there are 
# 11914 customer reviews in total.
# ---------------------------------------------------

from __future__ import division
from codecs import open

from collections import Counter
import numpy as np # In this assignment, probabilties will be calculated as logarithms.


def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')

split_point = int(0.80*len(all_docs)) # This will split the dataset into 80% and 20% sections, where the 80% is for training.
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:] 


# In this dataset, there are 9531 sets of sentences and corresponding class classifications.
def train_nb(documents, labels):
    # This counts how many of the sets of data are positive or negative. 
    class_counts = Counter(labels)

    # This counts the number of EACH word per class (positive or negative)
    word_counts = {"pos": Counter(), "neg": Counter()}

    # This counts how many words are in each class (positive or negative)
    total_word_counts = {"pos": 0, "neg": 0}


    vocabulary = set()
    index = 0
    for doc in documents:
        current_class = labels[index]
        for word in doc:
            vocabulary.add(word)

            # Count this word for its sentiment class (positive or negative)
            word_counts[current_class][word] += 1

            # Count total words for its sentiment class (positive or negative)
            total_word_counts[current_class] += 1
        index = index + 1
    
    # Counts the number of unique words        
    vocabulary_size = len(vocabulary)
    return class_counts, word_counts, total_word_counts, vocabulary_size


def score_doc_label(document, label, class_counts, word_counts, total_word_counts, vocabulary_size):

    # Calculates the prior term
    prob_of_class = class_counts[label] / sum(class_counts.values())
    log_prob_of_class = np.log(prob_of_class)


    # Calculates the likelihood term
    # If the word from the document is not present in the training data, this implementations provides smoothing via Laplace (aka Add 1)
    log_prob_words = 0
    for word in document:
        prob_word = (word_counts[label][word] + 1) / (total_word_counts[label] + vocabulary_size)
        log_prob_words += np.log(prob_word)

    # Returns the log of the probability of observing the words in a document, GIVEN a sentiment polarity label ("pos" or "neg")
    return log_prob_of_class + log_prob_words

# FOR ALL FOLLOWING SANITY CHECKS AND FUNCTION TESTS
class_counts, word_counts, total_word_counts, vocabulary_size = train_nb(train_docs, train_labels)

# FOR SANITY CHECK 1 (FROM ASSIGNMENT 4 INSTRUCTIONS)
print("Appearance of 'great' in the positive class: ", np.exp(score_doc_label(["great"],"pos", class_counts, word_counts, total_word_counts, vocabulary_size)))
print("Appearance of 'great' in the negative class: ", np.exp(score_doc_label(["great"],"neg", class_counts, word_counts, total_word_counts, vocabulary_size)))

print("Appearance of 'bad' in the positive class: ", np.exp(score_doc_label(["bad"],"pos", class_counts, word_counts, total_word_counts, vocabulary_size)))
print("Appearance of 'bad' in the negative class: ", np.exp(score_doc_label(["bad"],"neg", class_counts, word_counts, total_word_counts, vocabulary_size)))

# FOR SANITY CHECK 2 (FROM ASSIGNMENT 4 INSTRUCTIONS)
document = ['a', 'top-quality', 'performance']

# The following line returns a percent of 3.54e-10% for the positive class, INSTEAD OF CRASHING.
print("Probability of \"a top-quality performance\" being classified as positive:", np.exp(score_doc_label(document,"pos", class_counts, word_counts, total_word_counts, vocabulary_size)))

# The following line returns a percent of 1.81e-10% for the negative class, INSTEAD OF CRASHING.
print("Probability of \"a top-quality performance\" being classified as negative:", np.exp(score_doc_label(document,"neg", class_counts, word_counts, total_word_counts, vocabulary_size)))

def classify_nb(document, class_counts, word_counts, total_word_counts, vocabulary_size):
    class_guess = ""
    log_pos = score_doc_label(document, "pos", class_counts, word_counts, total_word_counts, vocabulary_size)
    log_neg = score_doc_label(document, "neg", class_counts, word_counts, total_word_counts, vocabulary_size)

    if log_pos > log_neg:
        class_guess = "pos"
    else: #The assumption here is to follow the negativity bias- if the probability is equal for it to be classified as positive or negative, it is assumed to be negative. That's just humans for you. 
        class_guess = "neg"

    return class_guess

document_test_1 = ['happy', 'and', 'soulful', 'it', 'is', 'a', 'masterpiece']
document_test_2 = ['kinda', 'boring', 'was', 'a', 'waste', 'of', 'money']

# FOR TESTING CLASSIFY_NB
print("Classification of Test 1: ", classify_nb(document_test_1, class_counts, word_counts, total_word_counts, vocabulary_size))
print("Classification of Test 2: ", classify_nb(document_test_2, class_counts, word_counts, total_word_counts, vocabulary_size))



def classify_documents(docs, class_counts, word_counts, total_word_counts, vocabulary_size):
    all_class_guesses = []

    for document in docs:
        all_class_guesses.append(classify_nb(document, class_counts, word_counts, total_word_counts, vocabulary_size))

    return all_class_guesses


def accuracy(true_labels, guessed_labels): 

    # Here, it is assumed that the number of the true labels equals the number of the guessed labels
    correct_guesses = 0
    incorrect_indices = []
    for i in range(len(true_labels)):
        if true_labels[i] == guessed_labels[i]:
            correct_guesses += 1
        else:
            incorrect_indices.append(i)
    

    return correct_guesses / (len(true_labels)), incorrect_indices

# Checking the accuracy of the guessed labels of the evaluation documents vs their true labels

guessed_labels = classify_documents(eval_docs, class_counts, word_counts, total_word_counts, vocabulary_size)
accuracy_decimal, incorrect_indices = accuracy(eval_labels, guessed_labels)
print("The accuracy of the Naive Bayes classification on this dataset is", accuracy_decimal*100,"%.")
print("Number of documents incorrectly classified:", len(incorrect_indices))


for i in range(int(len(incorrect_indices)/2)):
    document = eval_docs[incorrect_indices[i]]
    prob_pos = score_doc_label(document, "pos", class_counts, word_counts, total_word_counts, vocabulary_size)
    prob_neg = score_doc_label(document, "neg", class_counts, word_counts, total_word_counts, vocabulary_size)
    if (abs(prob_pos - prob_neg) > 5) & (len(document) < 50): 
        print(document)
        print("Guessed Classification:", classify_nb(document, class_counts, word_counts, total_word_counts, vocabulary_size))
        print("Estimated Probability of Positivity:", score_doc_label(document, "pos", class_counts, word_counts, total_word_counts, vocabulary_size))
        print("Estimated Probability of Negativity:", score_doc_label(document, "neg", class_counts, word_counts, total_word_counts, vocabulary_size))

    # The following yields a very LONG review being misclassified
    if (abs(prob_pos - prob_neg) > 90): 
            print(document)
            print(len(document))
            print("Guessed Classification:", classify_nb(document, class_counts, word_counts, total_word_counts, vocabulary_size))
            print("Estimated Probability of Positivity:", score_doc_label(document, "pos", class_counts, word_counts, total_word_counts, vocabulary_size))
            print("Estimated Probability of Negativity:", score_doc_label(document, "neg", class_counts, word_counts, total_word_counts, vocabulary_size))
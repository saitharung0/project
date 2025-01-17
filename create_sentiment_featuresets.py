import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
import numpy as np
import random
import pickle
from collections import Counter
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

def create_lexicon(pos,neg):

	lexicon = []
	with open(pos,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	with open(neg,'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			all_words = word_tokenize(l)
			lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	l2 = []
	for w in w_counts:
		#print(w_counts[w])
		if 1000 > w_counts[w] > 50:
			l2.append(w)
	print(len(l2))
	return l2





def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1

            featureset.append([features.tolist(), classification])

    return featureset



def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0])
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)

    # Check for consistent lengths
    feature_lengths = [len(feature[0]) for feature in features]
    unique_lengths = set(feature_lengths)
    print("Unique feature vector lengths:", unique_lengths)

    if len(unique_lengths) > 1:
        raise ValueError("Inconsistent feature vector lengths detected!")

    features = np.array(features, dtype=object)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels(r"C:\Users\Ravula Vineethreddy\Downloads\machine learning\pos.txt",r"C:\Users\Ravula Vineethreddy\Downloads\machine learning\neg.txt")
	# if you want to pickle this data:
	with open('C:/Users/Ravula Vineethreddy/Downloads/sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
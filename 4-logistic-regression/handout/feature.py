import csv
import numpy as np
import argparse
import time

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt

################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

class GloveEmbed:
    def __init__ (self, dict_file) :
        GloveEmbed.dictionary = load_feature_dictionary(dict_file)

class GloveEmbed_Data:
    def __init__(self, input_file, ge):
        # import and parse data
        self.data = np.genfromtxt(input_file, delimiter='\t', dtype='str')
        self.labels = self.data[:, 0].astype(int).reshape(-1, 1)
        self.reviews = np.char.split(self.data[:, 1]).reshape(-1, 1)

        # get feature vectors for all examples
        dictionary_embeddings = np.asarray(list(ge.dictionary.values()))
        self.review_embds = np.full((self.reviews.shape[0], len(list(ge.dictionary.values())[0])), -1.0)
        for idx, review in enumerate(self.reviews[:, 0]):
            # find word frequencies of dictionary in sentence
            words = np.array(review)
            mask = np.isin(words, list(ge.dictionary.keys()))
            vec = np.array([ge.dictionary[word] for word in words[mask]])

            # get feature vector for sentence
            self.review_embds[idx, :] = np.mean(vec, axis=0)

        # hstack labels for full feature vector
        self.review_embds = np.hstack((self.labels, self.review_embds))
        self.review_embds = np.around(self.review_embds, 6)

    def save_data(self, output_file):
        np.savetxt(output_file, self.review_embds, delimiter='\t', newline='\n', fmt='%.6f')

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, 
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    # train_input = sys.argv[1]
    # validation_input = sys.argv[2]
    # test_input = sys.argv[3]
    # glove_input = sys.argv[4]
    # train_out = sys.argv[5]
    # validation_out = sys.argv[6]
    # test_out = sys.argv[7]

    GloveEmbed_Data(args.train_input, GloveEmbed(args.feature_dictionary_in)).save_data(args.train_out)

    GloveEmbed_Data(args.validation_input, GloveEmbed(args.feature_dictionary_in)).save_data(args.validation_out)

    GloveEmbed_Data(args.test_input, GloveEmbed(args.feature_dictionary_in)).save_data(args.test_out)

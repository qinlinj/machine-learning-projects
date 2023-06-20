import argparse
import numpy as np


def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        train_data, words_to_index, tags_to_index, init_out, emit_out, trans_out = get_inputs()
    
    Where above the arguments have the following types:

        train_data --> A list of training examples, where each training example is a list
            of tuples train_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        init_out --> A file path to which you should write your initial probabilities

        emit_out --> A file path to which you should write your emission probabilities

        trans_out --> A file path to which you should write your transition probabilities
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmmprior", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)

    args = parser.parse_args()

    train_data = list()
    with open(args.train_input, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            train_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    return train_data, words_to_indices, tags_to_indices, args.hmmprior, args.hmmemit, args.hmmtrans

def initMatrix(tag_dict, tags):
    mat = np.ones(len(tag_dict))

    for seq in tags:
        idx = tag_dict.get(seq[0])
        mat[idx] += 1

    mat = mat.reshape(1, -1)
    return Normalize(mat)

def emissionMatrix(word_dict, tag_dict, sentences, tags):
    assert len(sentences) == len(tags)

    mat = np.ones((len(tag_dict), len(word_dict)))

    for (sent, seq) in zip(sentences, tags):
        for j in range(len(seq)):
            r = tag_dict.get(seq[j])
            c = word_dict.get(sent[j])
            mat[r, c] += 1

    return Normalize(mat)

def transitionMatrix(tag_dict, tags):
    mat = np.ones((len(tag_dict), len(tag_dict)))

    for seq in tags:
        for j in range(len(seq[:-1])):
            r = tag_dict.get(seq[j])
            c = tag_dict.get(seq[j + 1])
            mat[r, c] += 1

    return Normalize(mat)

def Normalize(mat):
    return mat / mat.sum(axis=1)[:, None]

if __name__ == "__main__":
    train_data, words_to_indices, tags_to_indices, init_out, emit_out, trans_out = get_inputs()

    sentences = [[word for word, _ in sentence_tag_pair] for sentence_tag_pair in train_data]
    tags = [[tag for _, tag in sentence_tag_pair] for sentence_tag_pair in train_data]

    init = initMatrix(tags_to_indices, tags)
    emission = emissionMatrix(words_to_indices, tags_to_indices, sentences, tags)
    transition = transitionMatrix(tags_to_indices, tags)

    np.savetxt(init_out, init, delimiter=" ")
    np.savetxt(emit_out, emission, delimiter=" ")
    np.savetxt(trans_out, transition, delimiter=" ")

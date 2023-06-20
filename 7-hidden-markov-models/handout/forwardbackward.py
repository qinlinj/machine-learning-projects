import argparse
import numpy as np

def get_inputs():
    """
    Collects all the inputs from the command line and returns the data. To use this function:

        validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = parse_args()

    Where above the arguments have the following types:

        validation_data --> A list of validation examples, where each element is a list:
            validation_data[i] = [(word1, tag1), (word2, tag2), (word3, tag3), ...]
        
        words_to_indices --> A dictionary mapping words to indices

        tags_to_indices --> A dictionary mapping tags to indices

        hmminit --> A np.ndarray matrix representing the initial probabilities

        hmmemit --> A np.ndarray matrix representing the emission probabilities

        hmmtrans --> A np.ndarray matrix representing the transition probabilities

        predicted_file --> A file path (string) to which you should write your predictions

        metric_file --> A file path (string) to which you should write your metrics
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("validation_data", type=str)
    parser.add_argument("index_to_word", type=str)
    parser.add_argument("index_to_tag", type=str)
    parser.add_argument("hmminit", type=str)
    parser.add_argument("hmmemit", type=str)
    parser.add_argument("hmmtrans", type=str)
    parser.add_argument("predicted_file", type=str)
    parser.add_argument("metric_file", type=str)

    args = parser.parse_args()

    validation_data = list()
    with open(args.validation_data, "r") as f:
        examples = f.read().strip().split("\n\n")
        for example in examples:
            xi = [pair.split("\t") for pair in example.split("\n")]
            validation_data.append(xi)
    
    with open(args.index_to_word, "r") as g:
        words_to_indices = {w: i for i, w in enumerate(g.read().strip().split("\n"))}
    
    with open(args.index_to_tag, "r") as h:
        tags_to_indices = {t: i for i, t in enumerate(h.read().strip().split("\n"))}
    
    hmminit = np.loadtxt(args.hmminit, dtype=float, delimiter=" ")
    hmmemit = np.loadtxt(args.hmmemit, dtype=float, delimiter=" ")
    hmmtrans = np.loadtxt(args.hmmtrans, dtype=float, delimiter=" ")

    return validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, args.predicted_file, args.metric_file

# You should implement a logsumexp function that takes in either a vector or matrix
# and performs the log-sum-exp trick on the vector, or on the rows of the matrix
def logsumexp(vec):
    m = np.max(vec)
    return m + np.log(np.sum(np.exp(vec - m)))

def forwardbackward(seq, loginit, logtrans, logemit, words_to_indices, tags_to_indices):
    """
       Your implementation of the forward-backward algorithm.

           seq is an input sequence, a list of words (represented as strings)

           loginit is a np.ndarray matrix containing the log of the initial matrix

           logtrans is a np.ndarray matrix containing the log of the transition matrix

           logemit is a np.ndarray matrix containing the log of the emission matrix

           words_to_indices --> A dictionary mapping words to indices

           tags_to_indices --> A dictionary mapping tags to indices

       You should compute the log-alpha and log-beta values and predict the tags for this sequence.
       """

    indices_to_tags = {index: tag for tag, index in tags_to_indices.items()}

    L = len(seq)
    M = len(loginit)

    log_alpha = np.zeros((L, M))
    log_beta = np.zeros((L, M))

    log_alpha[0] = loginit + logemit[:, words_to_indices[seq[0]]]

    for i in range(1, L):
        for j in range(M):
            log_alpha[i, j] = np.logaddexp.reduce(
                log_alpha[i - 1] + logtrans[:, j] + logemit[j, words_to_indices[seq[i]]])

    log_beta[-1] = np.zeros(M)

    for i in range(L - 2, -1, -1):
        for j in range(M):
            log_beta[i, j] = np.logaddexp.reduce(
                log_beta[i + 1] + logtrans[j, :] + logemit[:, words_to_indices[seq[i + 1]]])
    log_prob = logsumexp(log_alpha[-1])
    log_conditional_prob = log_alpha + log_beta - log_prob

    predicted_tags_indices = [np.argmax(log_conditional_prob[i]) for i in range(L)]

    predicted_tags = [indices_to_tags[index] for index in predicted_tags_indices]


    return predicted_tags, log_prob

if __name__ == "__main__":
    validation_data, words_to_indices, tags_to_indices, hmminit, hmmemit, hmmtrans, predicted_file, metric_file = get_inputs()

    total_tags = 0
    correct_tags = 0
    total_log_prob = 0.0
    num_examples = len(validation_data)
    with open(predicted_file, "w") as f:
        for example in validation_data:
            seq = [word for word, tag in example]
            true_tags = [tag for word, tag in example]

            loginit = np.log(hmminit)
            logtrans = np.log(hmmtrans)
            logemit = np.log(hmmemit)

            predicted_tags, log_prob = forwardbackward(seq, loginit, logtrans, logemit, words_to_indices,
                                                       tags_to_indices)

            total_tags += len(predicted_tags)

            correct_tags += sum(1 for pred, true in zip(predicted_tags, true_tags) if pred == true)
            total_log_prob += log_prob

            for i, (word, tag) in enumerate(example):
                f.write(f"{word}\t{predicted_tags[i]}\n")
            f.write("\n")

    accuracy = correct_tags / total_tags
    avg_log_likelihood = total_log_prob / len(validation_data)
    with open(metric_file, "w") as f:
        f.write(f"Average Log-Likelihood: {avg_log_likelihood:.16f}\n")
        f.write(f"Accuracy: {accuracy:.16f}\n")
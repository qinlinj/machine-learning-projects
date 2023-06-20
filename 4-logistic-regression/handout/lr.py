import numpy as np
import sys
import decimal
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (str): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def loadFormattedData(file):
    data = np.loadtxt(file, delimiter='\t', dtype=np.float16, comments=None, encoding='utf-8')
    labels = data[:,0].reshape(-1,1)
    intercept_features = np.ones(data.shape[0]).reshape(-1,1)
    data = np.hstack((labels, intercept_features, data[:,1:]))
    return labels, data


def updated_theta(theta, learning_rate, feature_vec):
    update = feature_vec[0] - sigmoid(np.dot(theta, feature_vec[1:]))
    theta = theta + learning_rate * update * feature_vec[1:]
    return theta


def calculate_nll(theta, feature_vec):
    nll = - feature_vec[0] * (np.dot(theta, feature_vec[1:])) + np.log(1 + np.exp(np.dot(theta, feature_vec[1:])))
    return nll


def train(train_file, valid_file, test_file, train_out, test_out, metrics_out,num_epoch=0, learning_rate=0.1, theta=None):

    train_labels, train_data = loadFormattedData(train_file)
    valid_labels, valid_data = loadFormattedData(valid_file)
    test_labels, test_data = loadFormattedData(test_file)


    if theta is None:
        theta = np.zeros(train_data.shape[1] - 1)


    predicted_labels_train = None
    predicted_labels_test = None
    predicted_train_error = 0.0
    predicted_test_error = 0.0

    all_train_nll = []
    all_valid_nll = []
    for itr in range(num_epoch):
        # update theta
        for feature_vec in train_data:
            theta = updated_theta(theta, learning_rate, feature_vec)
        print(itr)

        # train NLL
        total_train_nll = sum([calculate_nll(theta, feature_vec) for feature_vec in train_data])
        avg_train_nll = total_train_nll / train_data.shape[0]
        all_train_nll.append(avg_train_nll)

        # valid NLL
        total_valid_nll = sum([calculate_nll(theta, feature_vec) for feature_vec in valid_data])
        avg_valid_nll = total_valid_nll / valid_data.shape[0]
        all_valid_nll.append(avg_valid_nll)


    np.savetxt('train_NLL6.tsv', np.asarray(all_train_nll), delimiter='\t', newline='\n')
    np.savetxt('valid_NLL_model2.tsv', np.asarray(all_valid_nll), delimiter='\t', newline='\n')

    predicted_labels_train = sigmoid(np.dot(train_data[:, 1:], theta))
    predicted_labels_train = np.where(predicted_labels_train > 0.5, 1, 0).reshape(-1, 1)
    predicted_train_error = float(np.sum(predicted_labels_train != train_labels)) / float(
        predicted_labels_train.shape[0])

    predicted_labels_test = sigmoid(np.dot(test_data[:, 1:], theta))
    predicted_labels_test = np.where(predicted_labels_test > 0.5, 1, 0).reshape(-1, 1)
    predicted_test_error = np.sum(predicted_labels_test != test_labels, dtype=np.double) / predicted_labels_test.shape[
        0]

    train_label_file = open(train_out, 'w')
    test_label_file = open(test_out, 'w')
    metrics_file = open(metrics_out, 'w')

    for label in predicted_labels_train[:, 0]:
        train_label_file.write(str(label))
        train_label_file.write("\n")

    for label in predicted_labels_test[:, 0]:
        test_label_file.write(str(label))
        test_label_file.write("\n")

    metrics_file.write("error(train): ")
    metrics_file.write(str(predicted_train_error))
    metrics_file.write("\n")
    metrics_file.write("error(test): ")
    metrics_file.write(str(predicted_test_error))
    train_label_file.close()
    test_label_file.close()
    metrics_file.close()

if __name__ == '__main__' :
    # read arguments
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=str,
                        help='number of epochs of gradient descent to run')
    parser.add_argument("learning_rate", type=str,
                        help='learning rate for gradient descent')
    args = parser.parse_args()

    train(args.train_input, args.validation_input, args.test_input, args.train_out, args.test_out, args.metrics_out, int(args.num_epoch), float(args.learning_rate))
    print("finish")
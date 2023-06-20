import numpy as np
import sys

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("The input file is: %s" % (infile))
    print("The output file is: %s" % (outfile))

    # Reading the train input file
    train_data = np.genfromtxt(infile, delimiter='\t', skip_header=1)

    # Reading the test input file
    test_data = np.genfromtxt(outfile, delimiter='\t', skip_header=1)

    # Splitting the data into features and labels
    train_X = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_X = test_data[:, :-1]
    test_y = test_data[:, -1]

    # Initializing the majority voting classifier
    classifier = lambda x: np.round(np.mean(x))

    # Predicting the train labels
    train_pred = np.apply_along_axis(classifier, axis=1, arr=train_X)

    # Predicting the test labels
    test_pred = np.apply_along_axis(classifier, axis=1, arr=test_X)

    # Writing the train labels to the train out file
    np.savetxt(sys.argv[3], train_pred, fmt='%d')

    # Writing the test labels to the test out file
    np.savetxt(sys.argv[4], test_pred, fmt='%d')

    # Calculating the train and test error
    train_error = np.count_nonzero(train_pred - train_y) / len(train_y)
    test_error = np.count_nonzero(test_pred - test_y) / len(test_y)

    # Writing the train and test error to the metrics out file
    with open(sys.argv[5], 'w') as f:
        f.write('error(train): {}\n'.format(train_error))
        f.write('error(test): {}\n'.format(test_error))
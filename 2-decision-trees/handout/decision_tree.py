import sys
import csv
import numpy as np

def load_data(file_name):
    return np.genfromtxt(file_name, delimiter='\t', dtype=str, skip_header=1)


def read_first_line(file_name):
    try:
        fileHeaderArray = np.loadtxt(file_name, dtype=str, delimiter='\t')
        fileHeaderArray = fileHeaderArray[[0], ...]
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        return None
    return fileHeaderArray


def output_file(file_name, output_content):
    if isinstance(output_content, str):
        with open(file_name, 'w') as file_out:
            file_out.write(output_content)
    else:
        np.savetxt(file_name, output_content, fmt='%s', delimiter='\t', newline='\n')


def getEntropy(dataArrayY):
    data_val = np.unique(dataArrayY)

    data_size = len(dataArrayY)
    if len(data_val) == 1:
        return 0.0
    entropy = sum(
        -np.count_nonzero(dataArrayY == val) / data_size * np.log2(np.count_nonzero(dataArrayY == val) / data_size)
        for val in data_val)
    return entropy


def getMutualInfo(dataArrayY, dataArrayX, dataEntropyY=None):
    if dataEntropyY is None:
        dataEntropyY = getEntropy(dataArrayY)

    dataEntropyY_X = getCondEntropy(dataArrayY, dataArrayX)
    dataMutInfoYX = dataEntropyY - dataEntropyY_X
    return dataMutInfoYX

def getCondEntropy(dataArrayY, dataArrayX):
    dataSizeY = len(dataArrayY)
    dataSizeX = len(dataArrayX)
    dataValY = np.unique(dataArrayY)
    dataValX = np.unique(dataArrayX)

    if len(dataValY) == 1:
        return 0.0
    elif len(dataValX) == 1:
        return getEntropy(dataArrayY)
    else:
        dataEntropyY_X = sum(
            np.count_nonzero(dataArrayX == valX) / dataSizeX * getEntropy(
                dataArrayY[np.argwhere(dataArrayX == valX)])
            for valX in dataValX
        )
        return dataEntropyY_X


def majorityVote(input_data_array):
    label_idx = np.size(input_data_array, axis=1) - 1
    label_column = input_data_array[:, label_idx]
    unique_labels = np.unique(label_column)
    if len(unique_labels) == 2:
        count_first_label = np.count_nonzero(label_column == unique_labels[0])
        count_second_label = np.count_nonzero(label_column == unique_labels[1])
        if count_first_label > count_second_label:
            return unique_labels[0]
        elif count_first_label < count_second_label:
            return unique_labels[1]
        else:
            return sorted(unique_labels)[1]
    return unique_labels[0]


def errorRate(inputSet, predictLabel):
    try:
        labelIdx = np.size(inputSet, axis=1) - 1
        totalNum = np.size(inputSet, axis=0)
        realLabel = inputSet[:, labelIdx]
        erNum = 0
        for i in range(totalNum):
            if realLabel[i] != predictLabel[i]:
                erNum += 1
        erRate = erNum / totalNum
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    else:
        return erRate


def getSplitAttribute(inputDataArray):
    num_attributes = inputDataArray.shape[1] - 1
    mutual_info = np.zeros(num_attributes)
    label_data = inputDataArray[:, -1]
    for i in range(num_attributes):
        attr_data = inputDataArray[:, i]
        mutual_info[i] = getMutualInfo(label_data, attr_data)
    if mutual_info.max() == 0:
        return None
    return np.argmax(mutual_info)


def trainDecisionTree(inputDataArray, maxDepth):
    rootNode = dtNode(inputDataArray, 1, np.zeros((0, 1), dtype=str))

    def trainDecisionStump(inputNode):
        depthCheck = inputNode.depth > maxDepth
        if depthCheck:
            inputNode.predict = majorityVote(inputNode.data)
            return

        attributeCheck = (inputNode.attribute is None or len(inputNode.usedAttribute) == len(inputNode.data[0, :]))
        if attributeCheck:
            inputNode.predict = majorityVote(inputNode.data)
            return

        attributeArray = inputNode.data[..., [inputNode.attribute]]
        attributeVal = np.unique(attributeArray)
        inputNode.attributeVal = attributeVal

        for i in range(len(attributeVal)):
            if i == 0:
                nodeIdx = np.argwhere(attributeArray == attributeVal[i])
                nodeDataArray = inputNode.data[nodeIdx[..., 0]]
                nodeLabelVal = np.unique(nodeDataArray[..., -1])

                nodeLabelNum_1 = np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[0]) if len(
                    nodeLabelVal) == 2 else np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[0])
                nodeLabelNum_2 = np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[1]) if len(
                    nodeLabelVal) == 2 else 0

                inputNode.prtLNum1 = nodeLabelNum_1
                inputNode.prtLNum2 = nodeLabelNum_2

                inputNode.addLeft(nodeDataArray, inputNode.depth + 1, inputNode.usedAttribute)
                if not depthCheck:
                    trainDecisionStump(inputNode.left)
            else:
                nodeIdx = np.argwhere(attributeArray == attributeVal[i])
                nodeDataArray = inputNode.data[nodeIdx[..., 0]]
                nodeLabelVal = np.unique(nodeDataArray[..., -1])

                nodeLabelNum_1 = np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[0]) if len(
                    nodeLabelVal) == 2 else 0
                nodeLabelNum_2 = np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[0]) if len(
                    nodeLabelVal) == 2 else np.count_nonzero(nodeDataArray[..., [-1]] == nodeLabelVal[0])

                inputNode.prtRNum1 = nodeLabelNum_1
                inputNode.prtRNum2 = nodeLabelNum_2

                inputNode.addRight(nodeDataArray, inputNode.depth + 1, inputNode.usedAttribute)
                if not depthCheck:
                    trainDecisionStump(inputNode.right)

    trainDecisionStump(rootNode)
    return rootNode


def decisionStump(inputNode, inputDataRow):
    if inputNode.predict is not None:
        return inputNode.predict
    attributeIdx = inputNode.attribute
    nextNode = inputNode.left if inputDataRow[0][attributeIdx] == inputNode.attributeVal[0] else inputNode.right
    return decisionStump(nextNode, inputDataRow)


def predictLabel(inputNode, inputDataArray):
    predictNum = inputDataArray.shape[0]
    predictArray = np.empty((predictNum, 1), dtype=object)
    for i in range(predictNum):
        inputDataRow = inputDataArray[i:i + 1, :]
        predictArray[i] = decisionStump(inputNode, inputDataRow)
    return predictArray


def finalPrint(inputNode, fileHeaderArray, maxDepth):
    labelArray = inputNode.data[..., -1]
    labelVal = np.unique(labelArray)

    labelArray = inputNode.data[..., -1]
    uniqueLabels = np.unique(labelArray)
    classCount = []
    for i in range(len(uniqueLabels)):
        currentLabel = uniqueLabels[i]
        countOfCurrentLabel = np.count_nonzero(labelArray == currentLabel)
        classCount.append((currentLabel, countOfCurrentLabel))

    resultString = ""
    for i in range(len(classCount)):
        currentClass = classCount[i]
        resultString += f"{currentClass[1]} {currentClass[0]}/"

    print("\n[" + resultString[:-1] + "]")

    def drawRecur(inputNode, depth=0):
        if inputNode.depth > maxDepth:
            return

        if inputNode.attribute is not None:
            print(
                f"{'|' * depth} {fileHeaderArray[0, inputNode.attribute]} = {inputNode.attributeVal[0]}: [{inputNode.prtLNum1} {uniqueLabels[0]}/{inputNode.prtLNum2} {uniqueLabels[1]}]")
            drawRecur(inputNode.left, depth + 1)
            print(
                f"{'|' * depth} {fileHeaderArray[0, inputNode.attribute]} = {inputNode.attributeVal[1]}: [{inputNode.prtRNum1} {uniqueLabels[0]}/{inputNode.prtRNum2} {uniqueLabels[1]}]")
            drawRecur(inputNode.right, depth + 1)

    drawRecur(inputNode)


class dtNode(object):
    def __init__(self, nodeDataArray, nodeDepth, previousAttribute):
        self.data = nodeDataArray  # data: the data associated with this node
        self.depth = nodeDepth  # depth: the depth of this node in the tree
        self.dataNum = np.size(nodeDataArray, axis=0)  # dataNum: the number of instances in the data
        self.attribute = getSplitAttribute(self.data)  # attribute: the best attribute to split the data on
        self.attributeVal = None  # attributeVal: the possible values of the best attribute
        self.usedAttribute = np.append(previousAttribute, self.data[
            0, self.attribute])  # usedAttribute: the attributes used to split the data so far
        self.predict = None  # predict: the prediction for the instances in this node (majority vote of the labels in the data)
        self.prtLNum1 = 0  # prtLNum1: the number of instances with the first label in the left child node
        self.prtLNum2 = 0  # prtLNum2: the number of instances with the second label in the left child node
        self.prtRNum1 = 0  # prtRNum1: the number of instances with the first label in the right child node
        self.prtRNum2 = 0  # prtRNum2: the number of instances with the second label in the right child node
        self.left = None  # left: a reference to the left child node
        self.right = None  # right: a reference to the right child node

    def addLeft(self, leftDataArray, leftDepth, previousAttribute):
        self.left = dtNode(leftDataArray, leftDepth, previousAttribute)

    def addRight(self, rightDataArray, rightDepth, previousAttribute):
        self.right = dtNode(rightDataArray, rightDepth, previousAttribute)


def load_and_prepare_data(trainInput, testInput):
    train_data = load_data(trainInput)
    test_data = load_data(testInput)
    header = read_first_line(trainInput)
    return train_data, test_data, header


def train_and_predict(train_data, test_data, maxDepth):
    decision_tree = trainDecisionTree(train_data, maxDepth)
    train_prediction = predictLabel(decision_tree, train_data)
    test_prediction = predictLabel(decision_tree, test_data)
    return decision_tree, train_prediction, test_prediction


def calculate_metrics(train_data, train_prediction, test_data, test_prediction):
    train_error = errorRate(train_data, train_prediction)
    test_error = errorRate(test_data, test_prediction)
    metrics = f"error(train): {train_error}\nerror(test): {test_error}"
    return metrics


def output_results(trainOut, testOut, metricsOut, train_prediction, test_prediction, metrics):
    output_file(trainOut, train_prediction)
    output_file(testOut, test_prediction)
    output_file(metricsOut, metrics)

def main():
    trainInput = sys.argv[1]
    testInput = sys.argv[2]
    maxDepth = int(sys.argv[3])
    trainOut = sys.argv[4]
    testOut = sys.argv[5]
    metricsOut = sys.argv[6]
    train_data, test_data, header = load_and_prepare_data(trainInput, testInput)
    decision_tree, train_prediction, test_prediction = train_and_predict(train_data, test_data, maxDepth)
    metrics = calculate_metrics(train_data, train_prediction, test_data, test_prediction)
    output_results(trainOut, testOut, metricsOut, train_prediction, test_prediction, metrics)
    finalPrint(decision_tree, header, maxDepth)
    attributeNum = np.size(train_data, axis=1)
    print(attributeNum)

if __name__ == "__main__":
	main()
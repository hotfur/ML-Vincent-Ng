# HW2 Naive Bayesian Classifier
# Author: Vuong Kha Sieu
# Date: 22/10/2022

import numpy as np
import pandas as pd
import sys


def main():
    train = pd.read_table(sys.argv[1], sep='\t', lineterminator='\n')
    train_size = train.shape[0]
    train = train.groupby(list(train.columns)).size().reset_index(name="count")
    train_count = train["count"].sum()
    test = pd.read_table(sys.argv[2], sep='\t', lineterminator='\n')
    test_size = test.shape[0]
    test = test.groupby(list(test.columns)).size().reset_index(name="count")
    # Class probability
    root = {}
    for class_value, data in train.groupby("class", as_index=False):
        attr_list = {}
        data_count = data["count"].sum()
        attr_list["class_proba"] = round(data_count / train_count, 2)
        for attr in train.columns:
            if attr == "class" or attr == "count":
                continue
            attr_value_list = {}
            for attr_value, attr_split in data.groupby(attr, as_index=False):
                attr_value_list[attr_value] = round(attr_split["count"].sum()/data_count, 2)
            attr_list[attr] = attr_value_list
        root[class_value]=attr_list
    printer(root)
    train_acc = round(100 - (100/train_size)*(TestTree(data=train, tree=root)), 2)
    test_acc = round(100 - (100/test_size)*TestTree(data=test, tree=root), 2)
    print()
    print("Accuracy on training set (" + str(train_size) + " instances): " + str(train_acc) + "%")
    print()
    print("Accuracy on test set (" + str(test_size) + " instances): " + str(test_acc) + "%")


def printer(tree):
    for cls in tree:
        print("P(C=" + str(cls) + ")=" + str(tree[cls]["class_proba"]), end=" ")
        for attr in tree[cls]:
            if attr == "class_proba":
                continue
            for attr_value in tree[cls][attr]:
                print("P(" + str(attr) + "=" + str(attr_value) + "|" + str(cls) + ")=" + str(
                    tree[cls][attr][attr_value]), end=" ")
        print()

def TestTree(data, tree):
    """
    Classify the data based on a decision tree and output the number of incorrectly classified instances.
    """
    proba_arr_org = []
    for i in tree:
        proba_arr_org.append(tree[i]["class_proba"])
    incorrect = 0
    for row in data.index:
        proba_arr = list(proba_arr_org)
        for col in data.columns:
            if col == "class" or col == "count":
                continue
            for child in range(len(tree)):
                proba_arr[child] *= tree[child][col][data[col][row]]
        if data["class"][row] != np.argmax(proba_arr):
            incorrect += data["count"][row]
    return incorrect

if __name__ == "__main__":
    main()

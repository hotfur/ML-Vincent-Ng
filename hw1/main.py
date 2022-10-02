# HW1 Decision Tree
# Author: Vuong Kha Sieu
# Date: 27/09/2022

import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import multiprocessing
import sys
import tree


def main(percentage=1):
    train = pd.read_table(sys.argv[1], sep='\t', lineterminator='\n')
    test = pd.read_table(sys.argv[2], sep='\t', lineterminator='\n')
    # Split the train set for plotting accuracy
    mask = np.random.choice([True, False], size=train.shape[0], p=[percentage, 1-percentage])
    train = train[mask]
    # Tie breaking - Whole dataset
    global most_freq
    values, counts = np.unique(train["class"], return_counts=True)
    sort_values = np.argsort(-counts)
    most_freq = values[sort_values], counts[sort_values]
    # Precalculate the entropy for the whole dataset
    entropy = Entropy(counts)
    # Algorithm is destructive so keep a copy here for accuracy calculation later
    train_copy = train.copy(deep=True)
    root = LearnTree(train, entropy)
    train_acc = round(100 - (100/train_copy.shape[0])*(TestTree(data=train_copy, tree=root)), 1)
    test_acc = round(100 - (100/test.shape[0])*TestTree(data=test, tree=root), 1)
    return root, train_acc, test_acc, train_copy.shape[0], test.shape[0]


def LearnTree(data, entropy):
    """
    Build a decision tree based on training dataset and its entropy
    """
    # Base cases
    # Case 0: Entropy equals to 0 (pure node)
    if entropy == 0:
        return tree.Node(class_value=data["class"].iloc[-1])
    # Case 1: No more examples left
    if data.empty:
        return tree.Node(class_value=most_freq[0][0])
    # Case 2: Ran out of attributes for classification
    if len(data.columns) == 1:
        values, counts = np.unique(data["class"], return_counts=True)
        sort_values = np.argsort(-counts)
        local_most_freq = values[sort_values], counts[sort_values]
        # Tie breaking
        if local_most_freq[1][0] == local_most_freq[1][1]:
            value0 = local_most_freq[0][0]
            value1 = local_most_freq[0][1]
            if most_freq[1][most_freq[0] == value0] > most_freq[1][most_freq[0] == value1]:
                return tree.Node(class_value=local_most_freq[0][0])
            return tree.Node(class_value=local_most_freq[0][1])
        return tree.Node(class_value=local_most_freq[0][0])
    # Recursive case
    largest_IG = -9999
    attr_name = ""
    subnodes = list()
    subnodes_entropies = list()
    for attr in data.columns:
        # Skip class as it is not an attribute
        if attr == "class":
            continue
        # Split dataset by attribute into a list of n distinct values
        cur_subnodes = [y for x, y in data.groupby(attr, as_index=False)]
        # Entropy and Information Gain calculation for every possible subnodes
        cur_IG = 0
        cur_subnodes_entropies = list()
        for node in cur_subnodes:
            values, counts = np.unique(node["class"], return_counts=True)
            node_entropy = Entropy(counts)
            cur_subnodes_entropies.append(node_entropy)
            cur_IG -= node_entropy * node.shape[0]
        cur_IG = cur_IG/data.shape[0] + entropy
        if cur_IG > largest_IG:
            largest_IG = cur_IG
            attr_name = attr
            subnodes = cur_subnodes
            subnodes_entropies = cur_subnodes_entropies
    # Add the chosen attribute to tree datastructure
    cur_tree_children = list()
    attr_value_set = {0, 1, 2}
    for i in range(len(subnodes)):
        # Add attribute value to the subtree
        attr_value = subnodes[i][attr_name].iloc[-1]
        attr_value_set.remove(attr_value)
        # Drop the attribute from the subnode dataset
        subnodes[i].drop(attr_name, axis=1, inplace=True)
        subtree = LearnTree(subnodes[i], subnodes_entropies[i])
        # Assign the attribute name and value to the subtree
        subtree.attr = attr_name
        subtree.attr_value = attr_value
        # Add the subtree to current tree
        cur_tree_children.append(subtree)

    # Have to scan over all possible attribute values [0,1,2] and add to the tree
    # Can we save some data on the tree? They are going to base case 2 anyway...
    for i in attr_value_set:
        data.drop(data.index, inplace=True)
        subtree = LearnTree(data, 99)
        subtree.attr = attr_name
        subtree.attr_value = i
        cur_tree_children.append(subtree)
    return tree.Node(children=cur_tree_children)


def Entropy(count):
    """
    Calculate entropy from the distribution of a dataset
    """
    num_ins = np.sum(count)
    entropy = 0
    for i in count:
        if i == 0:
            continue
        entropy -= i * np.log2(i / num_ins)
    return entropy / num_ins


def TestTree(data, tree):
    """
    Classify the data based on a decision tree and output the number of incorrectly classified instances.
    """
    incorrect = 0
    if tree.attr is not None:
        if (tree.class_value is not None):
            incorrect += data.loc[(data["class"] != tree.class_value)].shape[0]
            return incorrect
    for child in tree.children:
        incorrect += TestTree(data[(data[child.attr] == child.attr_value)], child)
    return incorrect


def AccuracyTest(data):
    values, counts = np.unique(data["class"], return_counts=True)
    accuracy = 100/data.shape[0]
    for i in range(len(values)):
        if values[i] == 11:
            accuracy *= counts[i]
    return accuracy


def plotting(steps):
    """Dirty code"""
    acc_series = list()
    percent_dataset = np.linspace(start=1/steps, stop=1, num=steps)
    for i in percent_dataset:
        root, train_acc, test_acc, train_shape, test_shape = main(i)
        acc_series.append(test_acc)
    #plt.plot(percent_dataset, acc_series)
    #plt.show()

def plotting_multi(steps):
    """Dirty code"""
    pool = multiprocessing.Pool()
    acc_series = list()
    percent_dataset = np.linspace(start=1/steps, stop=1, num=steps)
    sessions = pool.map(main, percent_dataset)
    for session in sessions:
        acc_series.append(session[2])
    #plt.plot(percent_dataset, acc_series)
    #plt.show()

if __name__ == "__main__":
    #plotting_multi(100)
    root, train_acc, test_acc, train_shape, test_shape = main()
    root.printer()
    print()
    print("Accuracy on training set (" + str(train_shape) + " instances): " + str(train_acc) + "%")
    print()
    print("Accuracy on test set (" + str(test_shape) + " instances): " + str(test_acc) + "%")

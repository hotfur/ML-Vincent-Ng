# HW1 Decision Tree
# Author: Vuong Kha Sieu
# Date: 27/09/2022

import numpy as np
import pandas as pd
import sys
import tree


def main():
    train = pd.read_table(sys.argv[1], sep='\t', lineterminator='\n')
    test = pd.read_table(sys.argv[2], sep='\t', lineterminator='\n')
    # Tie breaking
    global most_freq
    values, counts = np.unique(train["class"], return_counts=True)
    sort_values = np.argsort(-counts)
    most_freq = values[sort_values], counts[sort_values]
    entropy = Entropy(counts)
    root = LearnTree(train, entropy)
    root.printer()
    print()
    train_acc = 100 - 100*(TestTree(data=train, tree=root))/train.shape[0]
    print("Accuracy on training set (" + str(train.shape[0]) + " instances): " + str(train_acc) + "%")
    print()
    test_acc = 100 - 100*TestTree(data=test, tree=root)/test.shape[0]
    print("Accuracy on test set (" + str(test.shape[0]) + " instances): " + str(test_acc) + "%")


def LearnTree(data, entropy):
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
        if attr == "class":
            continue
        cur_subnodes = [y for x, y in data.groupby(attr, as_index=False)]
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
    # Add to tree datastructure the result - Continue code here
    cur_tree_children = list()
    attr_value_set = {0, 1, 2}
    for i in range(len(subnodes)):
        # Drop the attribute from the subnode dataset
        attr_value = subnodes[i][attr_name].iloc[-1]
        attr_value_set.remove(attr_value)
        subnodes[i].drop(attr_name, axis=1, inplace=True)
        subtree = LearnTree(subnodes[i], subnodes_entropies[i])
        # Assign the attribute name and value to the subtree
        subtree.attr = attr_name
        subtree.attr_value = attr_value
        # Add the subtree to current tree
        cur_tree_children.append(subtree)

    # Trouble: Have to scan over all possible attribute values [0,1,2] but in reality there is no need for that
    for i in attr_value_set:
        data.drop(data.index, inplace=True)
        subtree = LearnTree(data, 99)
        subtree.attr = attr_name
        subtree.attr_value = i
        cur_tree_children.append(subtree)
    return tree.Node(children=cur_tree_children)


def Entropy(count):
    num_ins = np.sum(count)
    entropy = 0
    for i in count:
        if i == 0:
            continue
        entropy -= i * np.log2(i / num_ins)
    return entropy / num_ins


def TestTree(data, tree):
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


if __name__ == "__main__":
    main()

# HW1 Decision Tree
# Author: Vuong Kha Sieu
# Date: 27/09/2022

import numpy as np
import pandas as pd
import sys
import tree


def main():
    train = pd.read_table(sys.argv[1], sep='\t', lineterminator='\n')
    # test = pd.read_table(sys.argv[2], sep='\t', lineterminator='\n')
    # Tie breaking
    global most_freq
    count = np.bincount(train["class"])
    most_freq = np.argmax(count)
    entropy = Entropy(count)
    root = LearnTree(train, entropy)
    root.printer()

def LearnTree(data, entropy):
    # Base cases
    # Case 0: Entropy equals to 0 (pure node)
    if entropy == 0:
        return tree.Node(class_value=data["class"].iloc[-1])
    # Case 1: No more examples left
    if data.empty:
        return tree.Node(class_value=most_freq)
    # Case 2: Ran out of attributes for classification
    if len(data.columns) == 1:
        count = np.bincount(data["class"])
        max_index = np.argmax(count)
        if count[max_index] == count[most_freq]:
            return tree.Node(class_value=most_freq)
        return tree.Node(class_value=max_index)
    # Recursive case
    largest_IG = 0
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
            node_entropy = Entropy(np.bincount(node["class"]))
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
    for i in range(len(subnodes)):
        # Drop the attribute from the subnode dataset
        subnodes[i].drop(attr_name, axis=1, inplace=True)
        subtree = LearnTree(subnodes[i], subnodes_entropies[i])
        # Assign the attribute name and value to the subtree
        subtree.attr = attr_name
        subtree.attr_value = i
        # Add the subtree to current tree
        cur_tree_children.append(subtree)
    return tree.Node(children=cur_tree_children)


def Entropy(count):
    num_ins = np.sum(count)
    entropy = 0
    for i in count:
        if i == 0:
            continue
        entropy -= i * np.log2(i)
    return entropy / num_ins + np.log2(num_ins)


def TestTree():
    return


if __name__ == "__main__":
    main()

# A decision tree datastructure implementation
# Author: Vuong Kha Sieu

class Node:
    def __init__(self, children=None, attr=None, attr_value=None, class_value=None):
        """
        Create new tree node
        children: a list contains child treenodes
        attr: the attribute this treenode is classifing
        attr_value: the value of the attribute this treenode is classifing
        class_value: if this treenode is a leaf then it stores the resulting class value of the decision tree
        """
        self.children = children
        self.attr = attr
        self.attr_value = attr_value
        self.class_value = class_value

    # Insert Node
    def insert(self, attr=None, attr_value=None, class_value=None):
        """
        Built but never used
        Insert a new treenode into current treenode
        """
        if self.children is None:
            self.children = list()
        self.children.append(Node(attr=attr, attr_value=attr_value, class_value=class_value))

    # Print the Tree
    def printer(self, depth=-1):
        # Depth parameter added to prevent printing the root node, which has no attribute but has some children
        if depth != 0:
            for i in range(depth):
                print("| ", end="")
        if self.attr is not None:
            print(str(self.attr) + " = " + str(self.attr_value) + " :", end="")
            if self.class_value is not None:
                print(" " + str(self.class_value))
            else:
                print()

        if self.children is not None:
            for child in self.children:
                child.printer(depth=depth+1)

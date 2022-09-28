# A tree implementation
# Author: Vuong Kha Sieu

class Node:
    def __init__(self, children=None, attr=None, attr_value=None, class_value=None):
        self.children = children
        self.attr = attr
        self.attr_value = attr_value
        self.class_value = class_value

    # Insert Node
    def insert(self, attr=None, attr_value=None, class_value=None):
        if self.children is None:
            self.children = list()
        self.children.append(Node(attr=attr, attr_value=attr_value, class_value=class_value))

    # Print the Tree
    def printer(self, depth=-1):
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

    # Tree traversal for classification task
    def traversal(self, attr_list):
        # Base case
        if attr_list.empty:
            return self.class_value
        # Recursive case
        attr_list.remove(self.attr)
        return

import numpy as np
import math

class Tree_node:
    """
    Data structure for nodes in the decision-tree
    """
    def __init__(self,):
        self.feature = None # index of the selected feature (for non-leaf node)
        self.label = -1 # class label (for leaf node), -1 means the node is not a leaf node
        self.left_child = None # left child node
        self.right_child = None # right child node

class Decision_tree:
    """
    Decision tree with binary features
    """
    def __init__(self,min_entropy):
        self.min_entropy = min_entropy
        self.root = None
        # self.indexs = []

    def fit(self,train_x,train_y):
        # construct the decision-tree with recursion
        self.root = self.generate_tree(train_x,train_y)

    def predict(self,test_x):
        # iterate through all samples
        prediction = np.zeros([len(test_x),]).astype('int') # placeholder
        for i in range(len(test_x)):
            # traverse the decision-tree based on the features of the current sample till reaching a leaf node
            node = self.root
            while node.label == -1:
                if test_x[i][node.feature] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
            prediction[i] = node.label
        return prediction

    def generate_tree(self,data,label,indexs = []):
        # initialize the current tree node
        cur_node = Tree_node()

        # compute the node entropy
        node_entropy = self.compute_node_entropy(label)
        if indexs == []:
            indexs = [i for i in range(len(data[0]))]

        # determine if the current node is a leaf node based on minimum node entropy (if yes, find the corresponding class label with majority voting and exit the current recursion)
        if node_entropy < self.min_entropy:
            if len(label) == 0:
                cur_node.label  = 0
                return cur_node
            cur_node.label = np.argmax(np.bincount(label))
            return cur_node

        # select the feature that will best split the current non-leaf node
        selected_feature = self.select_feature(data,label)
        cur_node.feature = indexs[selected_feature]
        new_data = np.delete(data,selected_feature,axis=1)
        new_indexs = np.delete(indexs,selected_feature)

        # split the data based on the selected feature and start the next level of recursion
        left_y = []
        right_y = []
        left_data = []
        right_data = []
        for i in range(len(data)):
            if data[i][selected_feature] == 0:
                left_y.append(label[i])
                left_data.append(new_data[i])

            else:
                right_y.append(label[i])
                right_data.append(new_data[i])

        left_y = np.array(left_y)
        right_y = np.array(right_y)
        left_data = np.array(left_data)
        right_data = np.array(right_data)
        cur_node.left_child = self.generate_tree(left_data,left_y,new_indexs)
        cur_node.right_child = self.generate_tree(right_data,right_y,new_indexs)

        return cur_node

    def select_feature(self,data,label):
        # iterate through all features and compute their corresponding entropy
        best_feat = 0
        min_ent = 100

        for i in range(len(data[0])):
            # compute the entropy of splitting based on the selected features
            # ent_cur = self.compute_node_entropy(label)
            # left --> 0, right --> 1
            left_y = []
            right_y = []
            for j in range(len(data)):
                if data[j][i] == 0:
                    left_y.append(label[j])
                else:
                    right_y.append(label[j])
            split_ent = self.compute_split_entropy(left_y,right_y)
            # select the feature with minimum entropy
            if min_ent > split_ent:
                best_feat = i
                min_ent = split_ent

        return best_feat

    def compute_split_entropy(self,left_y,right_y):
        # compute the entropy of a potential split (with compute_node_entropy function), left_y and right_y are labels for the two branches
        split_entropy = 0 # placeholder
        ent_left = self.compute_node_entropy(left_y)
        ent_right = self.compute_node_entropy(right_y)
        counts = len(left_y) + len(right_y)
        split_entropy = (len(left_y)/counts) * ent_left + (len(right_y)/counts) * ent_right

        return split_entropy

    def compute_node_entropy(self,label):
        # compute the entropy of a tree node (add 1e-15 inside the log2 when computing the entropy to prevent numerical issue)
        node_entropy = 0 # placeholder
        counts = len(label)
        if counts == 0:
            return 0
        for i in range(10):
            p_i = np.sum(np.array(label)==i)/counts
            if p_i != 0:
                node_entropy -= (p_i)*math.log(p_i,2)
        return node_entropy

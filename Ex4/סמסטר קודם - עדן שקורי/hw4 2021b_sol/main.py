from binarytree import build
from Utilities import split_by_idx, calc_err_node, check_tag, pre_to_level,\
                        calc_err, calc_entropy, same_tag
import time
import copy

# Read the data into list of vectors and labels
vectors = []
labels = []

data = open("vectors.txt", "r")
for line in data:
    line = line.split()
    vectors.append(line[0:-1])
    labels.append(line[-1])
data.close()

vectors = [list(map(int, x)) for x in vectors]
labels = [list(map(int, x)) for x in labels]
labels = [item for sublist in labels for item in sublist]

# Collect k from user
k = int(input("Enter k:"))

# Because our exercise deal with vectors from {0,1}^8, the maximum k is 9
if k > 9:
    k = 9

# ##### PART A #######
"""
    Main recursively function of this section, check the min error of each sub-tree
    :return the tree that have the min error (pre-order)
"""


def brute_force(vec, lab, level):
    min_err = float('inf')
    best_sub_tree = None

    if level == 2:
        best_tag = [None, None]
        min_err_leaf = float('inf')
        min_i = None
        for i in range(8):
            pos, l_pos, neg, l_neg = split_by_idx(vec, i, lab)
            new_err_leaf = calc_err_node(l_pos) + calc_err_node(l_neg)
            if new_err_leaf < min_err_leaf:
                min_err_leaf = new_err_leaf
                best_tag[0] = check_tag(l_neg)
                best_tag[1] = check_tag(l_pos)
                min_i = i
        sub_tree = [min_i]
        sub_tree.extend(best_tag)
        return sub_tree

    for i in range(8):
        pos, l_pos, neg, l_neg = split_by_idx(vec, i, lab)
        # left son
        left_sub_tree = brute_force(neg, l_neg, level - 1)
        # right son
        right_sub_tree = brute_force(pos, l_pos, level - 1)

        curr = [i]
        curr.extend(left_sub_tree)
        curr.extend(right_sub_tree)
        curr_level_order = pre_to_level(curr)
        new_err = calc_err(vec, lab, curr_level_order)
        if new_err < min_err:
            min_err = new_err
            best_sub_tree = curr

    return best_sub_tree


# Run this section
startA= time.time()
ansA = brute_force(vectors, labels, k)
endA = time.time()
# Convert to level order
ansA = pre_to_level(ansA)

# Print error and draw the decision tree
binary_tree = build(ansA)
print('Part A-\nBinary tree from list for k = ' + str(k) + ':\n', binary_tree)

min_err_total = calc_err(vectors, labels, ansA)
print("The error is: " + str(min_err_total))
print("This section run in: " + str(endA - startA) + " seconds")

# ##### PART B #######

"""
    Main recursively function of this section, check the best split by index of each node (top to down)
    :return Decision tree that build by this strategy
"""

# initialization the parameters
lab = []
vec = []
dynasty = []
ansB = []


def min_entropy(vec, lab, was, level):
    new_was = copy.deepcopy(was)
    new_k = level - 1

    # Out of tree height
    if level == 0:
        return

    # leaf
    if level == 1:
        if vec is None or len(vec) == 0:
            ansB.append(None)
        else:
            ansB.append(check_tag(lab))
        return

    # This branch tagged before
    if vec is None:
        ansB.append(None)
        min_entropy(None, None, new_was, new_k)
        min_entropy(None, None, new_was, new_k)
        return

    # All vectors has same tag
    if same_tag(lab):
        ansB.append(check_tag(lab))
        min_entropy(None, None, new_was, new_k)
        min_entropy(None, None, new_was, new_k)
        return

    # decision node
    idx_min = None
    min_ent = float('inf')

    for i in range(8):
        if i not in new_was:
            new_ent = calc_entropy(vec, i, lab)
            if new_ent < min_ent:
                min_ent = new_ent
                idx_min = i

    pos, l_pos, neg, l_neg = split_by_idx(vec, idx_min, lab)
    new_was.append(idx_min)

    if len(pos) == 0 or len(neg) == 0:
        if len(pos) == 0:
            ansB.append(check_tag(l_neg))
        else:
            ansB.append((check_tag(l_pos)))
        min_entropy(None, None, new_was, new_k)
        min_entropy(None, None, new_was, new_k)
    else:
        ansB.append(idx_min)
        # Left son
        min_entropy(neg, l_neg, new_was, new_k)
        # Right son
        min_entropy(pos, l_pos, new_was, new_k)


# Run this section

startB = time.time()
min_entropy(vectors, labels, dynasty, k)
endB = time.time()

# Convert to level order
ansB = pre_to_level(ansB)

# Print error and draw the decision tree
binary_tree = build(ansB)
print('Part B-\nBinary tree from list for k = ' + str(k) + ':\n', binary_tree)

min_err_total = calc_err(vectors, labels, ansB)
print("The error is: " + str(min_err_total))
print("This section run in: " + str(endB - startB) + " seconds")

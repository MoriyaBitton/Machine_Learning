import math
import queue

"""
    Calculate the error of vectors and their labels on given list
    NOTE: the tree must be in level order! 
    :return error of the tree
"""


def calc_err(vectors, labels, ans):
    if len(labels)==0:
        return 0

    count = 0
    for vec, lab in zip(vectors, labels):
        curr = 0
        final_tag = None
        while final_tag is None:
            if ans[curr] is True or ans[curr] is False:
                final_tag = ans[curr]
            elif vec[ans[curr]] == 1:
                curr = curr * 2 + 2
            else:
                curr = curr * 2 + 1

        if final_tag is True and lab == 0:
            count = count + 1
        if final_tag is False and lab == 1:
            count = count + 1

    return count / len(labels)


"""
    Split the vectors and their labels by given index
    The positive side is the vectors that contain 1 in the index coordinate
    And the negative side is the vectors that contain 0 there.
"""


def split_by_idx(vectors, index, labels):
    pos = []
    neg = []
    l_pos = []
    l_neg = []
    for vec, l in zip(vectors, labels):
        if vec[index] == 1:
            pos.append(vec)
            l_pos.append(l)
        else:
            neg.append(vec)
            l_neg.append(l)
    return pos, l_pos, neg, l_neg


"""
    Calculate the entropy by parameter p
    NOTE:   This function undefined at 0 or 1,
            By the note from lee-ad: if p is 1 return 1, if p is 0 return 0.
    
"""


def entropy_func(p):
    if p == 0:
        return 0
    if p == 1:
        return 1
    else:
        return -p * (math.log(p, 2)) - ((1 - p) * math.log((1 - p), 2))



"""
    Return true if the given labels have more 1's then 0's. else false.
"""


def check_tag(labels):
    if len(labels) == 0:
        return None

    pos_count = 0
    neg_count = 0

    for i in labels:
        if i == 1:
            pos_count = pos_count + 1
        else:
            neg_count = neg_count + 1

    if pos_count > neg_count:
        return True
    return False


"""
    Calculate the entropy of vectors and labels that will be split by index.
"""


def calc_entropy(vectors, index, labels):
    pos, l_pos, neg, l_neg = split_by_idx(vectors, index, labels)

    count = 0
    for i in l_pos:
        if (i == 0):
            count = count + 1
    if len(pos) == 0:
        p_pos = 1
    else:
        if count == 0:
            p_pos = 0
        else:
            p_pos = count / len(l_pos)

    count = 0
    for i in l_neg:
        if (i == 0):
            count = count + 1
    if len(neg) == 0:
        p_neg = 1
    else:
        if count == 0:
            p_neg = 0
        else:
            p_neg = count / len(l_neg)

    entropy = entropy_func(p_pos) + entropy_func(p_neg)
    return entropy


"""
    Return True if all labels have same tag
"""


def same_tag(lab):
    if lab is None:
        print("labels are None")
        return

    if len(lab) == 0:
        print("labels are empty")
        return

    ans = lab[0]
    for i in range(len(lab)):
        if lab[i] != ans:
            return False
    return True


"""
    Return the minimum of sum of the 1's and sum of 0's
"""


def calc_err_node(labels):
    if len(labels) == 0:
        return 0
    pos_count = 0
    neg_count = 0

    for i in labels:
        if i == 1:
            pos_count = pos_count + 1
        else:
            neg_count = neg_count + 1

    return min(pos_count, neg_count)


"""
    Two functions from GeeksForGeeks to convert from pre-order of binary tree to level-order 
"""


def left_tree_size(n):
    if n <= 1: return 0
    l = int(math.log2(n + 1))  # l = no of completely filled levels
    ans = 2 ** (l - 1)
    last_level_nodes = min(n - 2 ** l + 1, ans)
    return ans + last_level_nodes - 1


def pre_to_level(arr):
    que = queue.Queue()
    que.put((0, len(arr)))
    res = []  # this will be answer
    while not que.empty():
        iroot, size = que.get()  # index of root and size of subtree
        if iroot >= len(arr) or size == 0:
            continue  ##nodes at iroot don't exist
        else:
            res.append(arr[iroot])  # append to back of output array
        sz_of_left = left_tree_size(size)
        que.put((iroot + 1, sz_of_left))  # insert left sub-tree info to que
        que.put((iroot + 1 + sz_of_left, size - sz_of_left - 1))  # right sub-tree info

    return res

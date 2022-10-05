from collections import Counter
import numpy as np
import torch
import itertools

class NodeTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return self.left, self.right

    def __str__(self):
        return self.left, self.right


def huffman_code_tree(node, binString=''):
    '''
    Function to find Huffman Code
    '''
    if type(node) in (np.float32, np.float64, np.int8, np.int64, str) :    # str for debugging
        return {node: binString}
    (l, r) = node.children()
    # import pdb; pdb.set_trace()
    d = dict()
    d.update(huffman_code_tree(l, binString + '0'))
    d.update(huffman_code_tree(r, binString + '1'))
    return d


def make_tree(nodes):
    '''
    Function to make tree
    :param nodes: Nodes
    :return: Root of the tree
    '''
    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))
        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
    return nodes[0][0]

def huff(rle):
    freq = dict(Counter(rle))
    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    node = make_tree(freq)
    encoding = huffman_code_tree(node)
    return encoding

def decode(root, enc):
    ret = []
    curr = root
    _len = len(enc)
    is_key = True
    key = 0
    for i in range(_len):
        if enc[i] == '0':
            curr = curr.left
        elif enc[i] == '1':
            curr = curr.right
        else:
            print(enc[i])
            raise NotImplementedError
        
        if (type(curr) in (np.int8, np.int64)):
            ret.append(curr)
            curr = root
    return np.array(ret)

def byte2bit(bytes):
    bit = []
    bytecode = bytes[:-2]
    for byte in bytecode:
        b = format(byte, '08b')
        bit.append(b)

    last_ele = format(bytes[-2], 'b')
    last_tar_len = bytes[-1]
    num_to_add_zeros = last_tar_len - len(last_ele)
    output =''.join(bit) + '0'*num_to_add_zeros + last_ele
    return output
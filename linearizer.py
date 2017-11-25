import numpy as np

operators = {'and': 0, 'or': 1, 'not': 2}
ope_li = ['and', 'or', 'not']


def linearize_tree(root, xp=np):
    # Left node indexes for all parent nodes
    lefts = []
    # Right node indexes for all parent nodes
    rights = []
    # Parent node indexes
    dests = []
    # All words of leaf nodes
    words = []

    # Current leaf node index
    n_leaf = 0

    def traverse_leaf(exp):
        nonlocal n_leaf
        if isinstance(exp, str) and exp in ope_li:
            leaf = operators[exp]
            words.append(leaf)
            n_leaf += 1
        elif isinstance(exp, str) and exp not in ope_li:
            leaf = int(exp) + len(operators)
            words.append(leaf)
            n_leaf += 1
        else:
            left, right = exp
            traverse_leaf(left)
            traverse_leaf(right)

    traverse_leaf(root)

    # Current internal node index
    node_index = n_leaf
    leaf_index = 0

    def traverse_node(exp):
        nonlocal leaf_index
        nonlocal node_index
        if isinstance(exp, str):
            leaf_index += 1
            return leaf_index - 1
        elif isinstance(exp, list):
            left, right = exp
            l = traverse_node(left)
            r = traverse_node(right)

            lefts.append(l)
            rights.append(r)
            dests.append(node_index)

            node_index += 1
            return node_index - 1

    traverse_node(root)
    assert len(lefts) == len(words) - 1

    """
    print("lefts : ", lefts)
    print("rights : ", rights)
    print("dests : ", dests)
    print("words : ", words)
    print("\n")
    """

    return {'lefts': xp.array(lefts, 'i'),
            'rights': xp.array(rights, 'i'),
            'dests': xp.array(dests, 'i'),
            'words': xp.array(words, 'i')
            }


def linearize_tree_relational(root, xp=np, n_var=6):
    # Left node indexes for all parent nodes
    lefts = []
    # Right node indexes for all parent nodes
    rights = []
    # Parent node indexes
    dests = []
    # Operations on each composition
    opes = []
    # All words of leaf nodes
    words = []

    # Current leaf node index
    n_leaf = 0

    def traverse_leaf(exp):
        nonlocal n_leaf
        if isinstance(exp, list) and exp[0] == 'not':
            leaf = int(exp[1]) + n_var
            words.append(leaf)
            n_leaf += 1

        elif isinstance(exp, str) and exp not in ope_li:
            leaf = int(exp)
            words.append(leaf)
            n_leaf += 1

        elif isinstance(exp, list):
            left, right = exp
            traverse_leaf(left)
            traverse_leaf(right)

    traverse_leaf(root)

    # Current internal node index
    node_index = n_leaf
    leaf_index = 0

    def traverse_node(exp):
        nonlocal leaf_index
        nonlocal node_index

        if isinstance(exp, list) and exp[0] == 'not':
            leaf_index += 1
            return leaf_index - 1
        elif isinstance(exp, str) and exp not in ope_li:
            leaf_index += 1
            return leaf_index - 1
        elif exp == 'and':
            return 'and'
        elif exp == 'or':
            return 'or'
        elif isinstance(exp, list):
            left, right = exp
            l = traverse_node(left)
            r = traverse_node(right)

            if l is 'and' or l is 'or':
                rights.append(r)
                if l is 'and':
                    opes.append(0)
                elif l is 'or':
                    opes.append(1)
                return node_index
            else:
                lefts.append(l)
                dests.append(node_index)
                node_index += 1
                return node_index - 1

    traverse_node(root)
    assert len(lefts) == len(words) - 1
    """
    print("lefts : ", lefts)
    print("rights : ", rights)
    print("dests : ", dests)
    print("words : ", words)
    print("opes: ", opes)
    print("\n")
    """
    return {'lefts': xp.array(lefts, 'i'),
            'rights': xp.array(rights, 'i'),
            'dests': xp.array(dests, 'i'),
            'words': xp.array(words, 'i'),
            'opes': xp.array(opes, 'i')
            }


# train, dev, test = reader.read()
# print(test[0][0])
# linearize_tree(test[0][0])

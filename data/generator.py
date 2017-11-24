from itertools import chain, combinations
from collections import Counter, defaultdict
import os
import random


def get_power_set(original_set):
    s = original_set
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def create_formula(universe, maxlen):

    def create_subtree(universe, maxlen, p_ope, right=True):
        if p_ope == form and right is True:
            operator = random.choice(operators)
        else:
            operator = random.choice(sub_operators[form])

        subtree = ()
        if operator == '0' or maxlen < 2:
            subtree = random.choice(list(universe))
            neg_or_none = random.choice(neg_or_nones)
            if neg_or_none == '0':
                return subtree
            else:
                return tuple([neg_or_none, subtree])

        else:
            lhs = create_subtree(universe, maxlen / 2, p_ope=operator, right=False)
            rhs = create_subtree(universe, maxlen / 2, p_ope=operator)
            subtree = tuple([lhs, operator, rhs])
        return subtree

    operators = ['and', 'or', '0']
    sub_operators = {'and': ['or', '0'], 'or': ['and', '0']}
    neg_or_nones = ['not', '0']
    form = random.choice(['and', 'or'])  # Choose cnf or dnf

    return create_subtree(universe, maxlen, p_ope=form)


def get_satisfying_worlds_for_tree(tree, power_set):
    if isinstance(tree, tuple):
        if tree[0] == 'not':
            child = get_satisfying_worlds_for_tree(tree[1], power_set)
            return power_set.difference(child)
        else:
            left = get_satisfying_worlds_for_tree(tree[0], power_set)
            right = get_satisfying_worlds_for_tree(tree[2], power_set)
            if tree[1] == "and":
                return left.intersection(right)
            elif tree[1] == "or":
                return left.union(right)
            else:
                print('syntax error', tree)
    else:
        result = []
        for world in power_set:
            if tree in world:
                result.append(world)
        return set(result)


def compute_relation(left, right, universe):
    ne_intersection = left.intersection(right)
    ne_just_left = left.difference(right)
    ne_just_right = right.difference(left)
    ne_outside = universe.difference(left.union(right))
    if ne_intersection and not ne_just_right and not ne_just_left and ne_outside:
        return "="
    elif ne_intersection and ne_just_right and not ne_just_left and ne_outside:
        return "<"
    elif ne_intersection and not ne_just_right and ne_just_left and ne_outside:
        return ">"
    elif not ne_intersection and ne_just_right and ne_just_left and not ne_outside:
        return "^"
    elif not ne_intersection and ne_just_right and ne_just_left and ne_outside:
        return "|"
    elif ne_intersection and ne_just_right and ne_just_left and not ne_outside:
        return "v"
    else:
        return "#"


def get_len(tree):
    if isinstance(tree, tuple):
        accum = 0
        for entry in tree:
            accum += get_len(entry)
        return accum
    elif tree == 'and' or tree == 'or' or tree == 'not':
        return 1
    else:
        return 0


def to_string(expr):
    if isinstance(expr, int) or isinstance(expr, str):
        return str(expr)
    elif len(expr) == 2:
        return "( " + to_string(expr[0]) + " " + to_string(expr[1]) + " )"
    elif len(expr) == 3:
        return "( " + to_string(expr[0]) + " ( " + to_string(expr[1]) + " " + to_string(expr[2]) + " ) )"


def uniq(seq, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


def main(n_var, max_n_var, max_n_operator, train_portion):
    original_set = set(range(n_var))
    power_set = set(get_power_set(original_set))

    stats = Counter()
    total = 0

    outputs = defaultdict(list)

    while total < 500000:
        subset = random.sample(original_set, max_n_var)
        l_formula = create_formula(subset, max_n_operator)
        r_formula = create_formula(subset, max_n_operator)
        satl = get_satisfying_worlds_for_tree(l_formula, power_set)
        satr = get_satisfying_worlds_for_tree(r_formula, power_set)
        if satl == power_set or len(satl) == 0:
            continue
        if satr == power_set or len(satl) == 0:
            continue

        rel = compute_relation(satl, satr, power_set)

        if rel != "?":
            stats[rel] += 1
            total += 1
            max_len = min(max(get_len(l_formula), get_len(r_formula)), 12)
            outputs[max_len].append("" + rel + "\t" + to_string(l_formula) + "\t" + to_string(r_formula))

    to_dev = train_portion + (1 - train_portion) / 2
    dir_name = '{}_{}'.format(N_VAR, MAX_N_VAR)
    os.mkdir(dir_name)
    for length in outputs.keys():
        outputs[length] = uniq(outputs[length])

        print('length:', length)
        print('n:', len(outputs[length]))
        filename = dir_name + '/train' + str(length)
        f = open(filename, 'w')
        for i in range(int(train_portion * len(outputs[length]))):
            output = outputs[length][i]
            f.write(output + "\n")
        f.close()

        filename = dir_name + '/dev' + str(length)
        f = open(filename, 'w')
        for i in range(int(train_portion * len(outputs[length])), int(len(outputs[length]) * to_dev)):
            output = outputs[length][i]
            f.write(output + "\n")
        f.close()

        filename = dir_name + '/test' + str(length)
        f = open(filename, 'w')
        for i in range(int(len(outputs[length]) * to_dev), len(outputs[length])):
            output = outputs[length][i]
            f.write(output + "\n")
        f.close()

    print(stats)
    return None


N_VAR = 6
MAX_N_VAR = 4
MAX_N_OPERATOR = 12
TRAIN_PORTION = 0.7
if __name__ == '__main__':
    main(N_VAR, MAX_N_VAR, MAX_N_OPERATOR, TRAIN_PORTION)

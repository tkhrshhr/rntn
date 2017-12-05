relations = {'=': 0, '<': 1, '>': 2, '^': 3, '|': 4, 'v': 5, '#': 6}


class TreeParser():

    def __init__(self):
        self.root = None
        self._stack = [[]]

    def parse(self, tree_string):
        if len(tree_string) == 1:
            return tree_string

        reading = []
        for character in tree_string.strip():
            if character == '(':
                self._stack.append([])
            elif character == ' ':
                if reading:
                    self._stack[-1].append(''.join(reading))
                    reading.clear()
            elif character == ')':
                if reading:
                    self._stack[-1].append(''.join(reading))
                reading.clear()
                self._stack[-2].append(self._stack.pop())
            else:  # string
                reading.append(character)
        self.root = self._stack.pop()

        self._stack = [[]]
        return self.root[0]


def read_each_file(file_object):
    parser = TreeParser()
    s_i = []
    o_i = []
    r_i = []

    def get_example(line):
        r, s, o = line.strip('\n').split('\t')

        s_parsed = parser.parse(s)
        o_parsed = parser.parse(o)
        r_id = relations[r]

        return s_parsed, o_parsed, r_id

    for line in file_object.readlines():
        s, o, r = get_example(line)
        s_i.append(s)
        o_i.append(o)
        r_i.append(r)

    return s_i, o_i, r_i


def read(n_var=6, max_n_var=4):
    dir_name = '{}_{}'.format(n_var, max_n_var)

    def read_each_data(data_name, split1, split2):
        s = []
        o = []
        r = []
        for i in range(split1, split2):
            with open('data/{}/{}{}'.format(dir_name, data_name, i), 'r') as f:
                s_i, o_i, r_i = read_each_file(f)
                s += s_i
                o += o_i
                r += r_i
        return s, o, r

    # Train
    train = read_each_data('train', 0, 5)
    # Dev
    dev = read_each_data('dev', 0, 5)
    # Test
    test = []
    for i in range(13):
        bin = read_each_data('test', i, i + 1)
        test.append(bin)

    return train, dev, test


read()

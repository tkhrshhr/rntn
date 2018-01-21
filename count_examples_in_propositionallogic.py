train = 0
for i in range(0, 5):
    with open('data/6_4/train{}'.format(i), 'r') as f:
        train += len(f.readlines())
dev = 0
n_sharp_dev = 0
for i in range(0, 5):
    with open('data/6_4/dev{}'.format(i), 'r') as f:
        dev += len(f.readlines())
        f.seek(0)
        for line in f.readlines():
            r = line[0]
            if r == '#':
                n_sharp_dev += 1

print("n_sharp_dev:", n_sharp_dev / dev)

test = 0
for i in range(0, 13):
    with open('data/6_4//test{}'.format(i), 'r') as f:
        test += len(f.readlines())
        f.seek(0)
        total = len(f.readlines())
        f.seek(0)
        n_sharp = 0
        for line in f.readlines():
            r = line[0]
            if r == '#':
                n_sharp += 1
        print(n_sharp / total)
print(train)
print(dev)
print(test)

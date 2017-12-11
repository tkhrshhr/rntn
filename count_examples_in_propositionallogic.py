train = 0
for i in range(0, 5):
    with open('data/6_4/train{}'.format(i), 'r') as f:
        train += len(f.readlines())
dev = 0
for i in range(0, 5):
    with open('data/6_4/dev{}'.format(i), 'r') as f:
        dev += len(f.readlines())
test = 0
for i in range(0, 5):
    with open('data/6_4//test{}'.format(i), 'r') as f:
        test += len(f.readlines())
print(train)
print(dev)
print(test)

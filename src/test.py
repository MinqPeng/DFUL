from collections import defaultdict

Pred = defaultdict(list)


def test_f(Pred, n):
    if n == 0:
        Pred['0'] += [1,2]
    if n == 1:
        Pred['1'] += [8,7]
    if n == 2:
        Pred['0'] += [3,4]
    if n == 3:
        Pred['1'] += [6,5]
    if n == 4:
        Pred['0'] += [5,6]
    if n == 4:
        Pred['1'] += [4,3]
        
for i in range(6):
    test_f(Pred, i)
print(Pred)


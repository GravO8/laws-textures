import numpy as np

vectors = {
    "L5": np.array([[ 1, 4, 6, 4, 1]]),
    "E5": np.array([[-1,-2, 0, 2, 1]]),
    "S5": np.array([[-1, 0, 2, 0,-1]]),
    # "W5": np.array([[-1, 2, 0,-2, 1]]),
    "R5": np.array([[1, -4, 6,-4, 1]])
}

'''
   L5 E5 S5 R5
L5     1  6  2
E5     7  3  8
S5        4  9
R5           5

1  L5E5/E5L5
2  L5R5/R5L5
3  E5S5/S5E5
4  S5S5
5  R5R5
6  L5S5/S5L5
7  E5E5
8  E5R5/R5E5
9  S5R5/R5S5
'''


kernels = {}
for v1 in vectors:
    for v2 in vectors:
        name            = v1 + v2
        kernel          = vectors[v1].T * vectors[v2]
        kernels[name]   = kernel
        # print(kernel)
        # print()

seen = {}
i = 0
for k in kernels:
    for s in seen:
        if (seen[s].T == kernels[k]).all():
            i += 1
            print(i, f"{s}/{k}")
            break
    seen[k] = kernels[k]
print(len(kernels)-i)

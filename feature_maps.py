import numpy as np, itertools
from scipy import signal

VECTOR_INITIALS = ["L", "E", "S", "W", "R", "O"]
'''
L - Level
E - Edge
S - Spot
R - Ripple
W - wave
'''


class FeatureMaps:
    def __init__(self, basis_vectors: list, vector_dims: int):
        assert vector_dims > 1
        assert vector_dims % 2 == 1
        for v in basis_vectors:
            assert (v.ndim == 1) and (len(v) == 3)
        self.basis_vectors  = basis_vectors
        self.vector_dims    = vector_dims
        self.vec_counter    = 0
        self.generate_vectors()
        
    def get_vector_name(self, j):
        self.vec_counter += 1
        return f"X{self.vec_counter}-" + str(self.vector_dims)

    def generate_vectors(self):
        N            = 1 + (self.vector_dims - 3) // 2
        j            = 0
        combinations = list(itertools.combinations_with_replacement(range(len(self.basis_vectors)), N))
        # self.vectors = {}
        self.vectors = []
        for comb in combinations:
            vector = self.basis_vectors[comb[0]]
            for i in range(1,len(comb)):
                vector = np.convolve(vector, self.basis_vectors[comb[i]])
            # self.vectors[self.get_vector_name(j)] = vector
            self.vectors.append(vector)
            j += 1
            
    def get_kernel_shape(self, input_dim, dim):
        return tuple([self.vector_dims if i == dim else 1 for i in range(input_dim)])
            
    def get_features(self, x, preprocess: bool = True, merge_symmetric: bool = True):
        input_dim = x.ndim
        kernels_i = list(itertools.combinations_with_replacement(range(self.vector_dims+1), input_dim))
        n_maps    = len(self.vectors)**(input_dim) -1
        permutations = []
        for kernel_i in kernels_i:
            permutations.append( list(set(itertools.permutations(kernel_i, input_dim))) )
        permutations.pop(0) # remove the first non zero sum kernel
        # if preprocess:
            # TODO
        maps = np.repeat(x[..., np.newaxis], n_maps, axis = input_dim)
        for dim in range(input_dim):
            shape = self.get_kernel_shape(input_dim, dim)
            for permutation_set in permutations:
                for kernel_i in permutation_set: 
                    i = kernel_i[dim]
                    maps[..., dim] = signal.fftconvolve(maps[..., dim], self.vectors[i].reshape(shape), mode = "same")            
        if merge_symmetric:
            feature_indexes = []
            i = 0
            j = 0
            for permutation_set in permutations:
                n_perm = len(permutation_set)
                maps[...,j] = maps[..., i:i+n_perm].mean()
                i += n_perm
                j += 1
            assert j == len(kernels_i)-1
            maps = maps[...,:j]
        return maps

def laws_textures(vector_dims: int = 5):
    basis_vectors = [
        np.array([ 1, 2, 1]),
        np.array([-1, 0, 1]),
        np.array([-1, 2,-1])
    ]
    return FeatureMaps(basis_vectors, vector_dims)


if __name__ == "__main__":
    laws = laws_textures(vector_dims = 5)
    x = np.zeros((32,32,32))
    laws.get_features(x)

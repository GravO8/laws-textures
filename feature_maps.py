import numpy as np, itertools
from abc import ABC
from scipy import signal

VECTOR_INITIALS = ["L", "E", "S", "W", "R", "O"]
'''
L - Level
E - Edge
S - Spot
R - Ripple
W - wave
'''
LAWS_VECTORS = {
    3: [np.array([ 1, 2, 1]),  # L3
        np.array([-1, 0, 1]),  # E3
        np.array([-1, 2,-1])], # S3
    5: [np.array([ 1, 4, 6, 4, 1]),  # L5
        np.array([-1,-2, 0, 2, 1]),  # E5
        np.array([-1, 0, 2, 0,-1]),  # S5
        np.array([-1, 2, 0,-2, 1]),  # W5
        np.array([ 1,-4, 6,-4, 1])], # R5
    7: [np.array([ 1, 6,15,20,15, 6, 1]), # L7
        np.array([-1,-4,-5, 0, 5, 4, 1]), # E7
        np.array([-1,-2, 1, 4, 1,-2,-1]), # S7
        np.array([-1, 0, 3, 0,-3, 0, 1]), # W7
        np.array([ 1,-2,-1, 4,-1,-2, 1]), # R7
        np.array([-1,6,-15,20,-15,6,-1])] # O7
}

class FeatureMaps(ABC):
    def __init__(self, vectors):
        self.vectors = vectors
        self.vector_dims = len(vectors[0])
        for v in self.vectors[1:]:
            assert len(v) == self.vector_dims
    
    def get_kernel_shape(self, input_dim, dim):
        return tuple([self.vector_dims if i == dim else 1 for i in range(input_dim)])
        
    def preprocess_input(self, x, window_size, input_dim):
        smooth   = np.ones((window_size,)*input_dim)/(window_size**2)
        x        = signal.fftconvolve(x, smooth, mode = "same")
        return x
        
    def merge_symmetric_maps(self, maps, permutations):
        i = 0
        j = 0
        for permutation_set in permutations:
            n_perm = len(permutation_set)
            maps[...,j] = np.mean(maps[..., i:i+n_perm], axis = -1)
            i += n_perm
            j += 1
        maps = maps[...,:j]
        return maps
        
    def compute_energy_maps(self, maps, window_size, input_dim):
        maps     = np.abs(maps)
        abs_save = np.ones((window_size,)*input_dim) # absolute save
        maps     = np.stack([signal.fftconvolve(maps[...,i], abs_save, mode = "valid") for i in range(maps.shape[-1])], axis = -1)
        return maps
        
    def generate_kernel_permutations(self, input_dim):
        kernels_i    = list(itertools.combinations_with_replacement(range(len(self.vectors)), input_dim))
        permutations = []
        for kernel_i in kernels_i:
            permutations.append( list(set(itertools.permutations(kernel_i, input_dim))) )
        permutations.pop(0) # remove the first non zero sum kernel
        return permutations
        
    def compute_maps_separably(self, x, input_dim, permutations):
        n_maps = len(self.vectors)**(input_dim) -1
        maps   = np.repeat(x[..., np.newaxis], n_maps, axis = input_dim)
        j      = 0
        for permutation_set in permutations:
            for kernel_i in permutation_set:
                for dim in range(input_dim):
                    shape = self.get_kernel_shape(input_dim, dim)
                    maps[...,j] = signal.fftconvolve(maps[...,j], 
                                                    self.vectors[kernel_i[dim]].reshape(shape), 
                                                    mode = "same")
                j += 1
        return maps
        
    def compute_maps_fully(self, x, input_dim, permutations):
        n_maps = len(self.vectors)**(input_dim) -1
        maps   = np.repeat(x[..., np.newaxis], n_maps, axis = input_dim)
        j      = 0
        for permutation_set in permutations:
            for kernel_i in permutation_set:
                shape  = self.get_kernel_shape(input_dim, 0)
                kernel = self.vectors[kernel_i[0]].reshape(shape).copy()
                for dim in range(1,input_dim):
                    shape   = self.get_kernel_shape(input_dim, dim)
                    kernel = kernel * self.vectors[kernel_i[0]].reshape(shape)
                maps[...,j] = signal.fftconvolve(maps[...,j], 
                                                kernel, 
                                                mode = "same")
                j += 1
        return maps
        
    def get_features(self, x, window_size: int = 15, preprocess: bool = True, 
        merge_symmetric: bool = True, compute_energy: bool = True, 
        compute_fully: bool = False):
        input_dim = x.ndim
        if preprocess:
            x = self.preprocess_input(x, window_size, input_dim)
        permutations = self.generate_kernel_permutations(input_dim)
        if compute_fully:
            maps = self.compute_maps_fully(x, input_dim, permutations)
        else:
            maps = self.compute_maps_separably(x, input_dim, permutations)
        if merge_symmetric:
            maps = self.merge_symmetric_maps(maps, permutations)
        if compute_energy:
            maps = self.compute_energy_maps(maps, window_size, input_dim)
        return maps
        
        
class GeneralizedFeatureMaps(FeatureMaps):
    def __init__(self, basis_vectors: list, vector_dims: int):
        assert vector_dims > 1
        assert vector_dims % 2 == 1
        for v in basis_vectors:
            assert (v.ndim == 1) and (len(v) == 3)
        self.basis_vectors  = basis_vectors
        self.vec_counter    = 0
        vectors             = self.generate_vectors(vector_dims)
        super().__init__(vectors)
        
    def get_vector_name(self, j):
        self.vec_counter += 1
        return f"X{self.vec_counter}-" + str(self.vector_dims)

    def generate_vectors(self, vector_dims):
        N            = 1 + (vector_dims - 3) // 2
        j            = 0
        combinations = list(itertools.combinations_with_replacement(range(len(self.basis_vectors)), N))
        vectors = []
        for comb in combinations:
            vector = self.basis_vectors[comb[0]]
            for i in range(1,len(comb)):
                vector = np.convolve(vector, self.basis_vectors[comb[i]])
            vectors.append(vector)
            j += 1
        return vectors


def laws_textures(vector_dims: int = 5):
    assert vector_dims in LAWS_VECTORS
    return FeatureMaps(LAWS_VECTORS[vector_dims])


if __name__ == "__main__":
    laws = laws_textures(vector_dims = 5)
    x = np.zeros((32,32))
    laws.get_features(x, compute_fully = True)
    
    # maps = GeneralizedFeatureMaps(LAWS_VECTORS[3], 5)
    # maps.get_features(x)

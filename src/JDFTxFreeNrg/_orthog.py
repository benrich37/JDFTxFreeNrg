import numpy as np
from JDFTxFreeNrg._common import dagger, invsqrt, _error_on_nan_in_list_of_floats, _error_on_nan_in_list_of_vectors, _error_on_nan_in_array

"""
Collections of functions to orthogonalize sets of vectors and projectors
"""



def remove_parallel_vectors(vectors: list[np.ndarray], cutoff: float = 1e-4) -> list[np.ndarray]:
    try:
        vectors = _remove_parallel_vectors(vectors, cutoff=cutoff)
        _error_on_nan_in_list_of_vectors(vectors, context="after running remove_parallel_vectors")
        return vectors
    except Exception as e:
        print(f"Error in removing parallel vectors")
        raise e

def _remove_parallel_vectors(vectors: list[np.ndarray], cutoff: float = 1e-4) -> list[np.ndarray]:
    parallel_pairs = []
    for i, vec in enumerate(vectors):
        overlaps = [abs(vec @ v) for v in vectors]
        _error_on_nan_in_list_of_floats(overlaps, context=f"overlaps of vector index {i} with others")
        for j, o in enumerate(overlaps):
            if i != j and abs(o - 1) < cutoff:
                parallel_pairs.append((min(i, j), max(i, j)))
    parallel_pairs = list(set(parallel_pairs))
    # print(parallel_pairs)
    pop_idcs = sorted(set([j for i, j in parallel_pairs]), reverse=True)
    # print(pop_idcs)
    for idx in pop_idcs:
        vectors.pop(idx)
    return vectors

def orthogonalize_vector_set_step(vectors: list[np.ndarray], idx: int, cutoff: float = 1e-4) -> list[np.ndarray]:
    try:
        vectors = _orthogonalize_vector_set_step(vectors, idx, cutoff=cutoff)
        _error_on_nan_in_list_of_vectors(vectors, context=f"after orthogonalizing vector set at index {idx}")
        return vectors
    except Exception as e:
        print(f"Error in orthogonalizing vector set at index {idx}")
        raise e

def _orthogonalize_vector_set_step(vectors: list[np.ndarray], idx: int, cutoff: float = 1e-4) -> list[np.ndarray]:
    v1 = vectors[idx] / np.linalg.norm(vectors[idx])
    new_vectors = []
    for j, vec in enumerate(vectors):
        if j == idx:
            new_vectors.append(v1)
            continue
        v2 = vec / np.linalg.norm(vec)
        overlap = np.dot(v1, v2)
        if abs(overlap) > 1 - cutoff:
            continue
        orth_v2 = v2 - overlap * v1
        new_vectors.append(orth_v2 / np.linalg.norm(orth_v2))
    return new_vectors

def remove_parallel_vectors_loop(vectors: list[np.ndarray], cutoff: float = 1e-4, maxstep: int = 10) -> list[np.ndarray]:
    try:
        vectors = _remove_parallel_vectors_loop(vectors, cutoff=cutoff, maxstep=maxstep)
        _error_on_nan_in_list_of_vectors(vectors, context="after running remove_parallel_vectors_loop")
        return vectors
    except Exception as e:
        print(f"Error in running remove_parallel_vectors_loop")
        raise e

def _remove_parallel_vectors_loop(vectors: list[np.ndarray], cutoff: float = 1e-4, maxstep: int = 10) -> list[np.ndarray]:
    step = 0
    _vectors = vectors
    while step < maxstep:
        #print("step ", step)
        initial_len = len(_vectors)
        #print(f"  Initial number of vectors: {initial_len}")
        for i in range(len(_vectors)):
            #print("substep ", i)
            _vectors = remove_parallel_vectors(_vectors, cutoff=cutoff)
            if len(_vectors) != initial_len:
                #print(f"  Number of vectors after removal: {len(_vectors)}")
                break
            _vectors = orthogonalize_vector_set_step(_vectors, i, cutoff=cutoff)
            if len(_vectors) != initial_len:
                #print(f"  Number of vectors after orthogonalization: {len(_vectors)}")
                break
        step += 1
    return _vectors


def orthogonalize_projector(projector1):
    try:
        projector = _orthogonalize_projector(projector1)
        _error_on_nan_in_array(projector, context="after orthogonalizing projector")
        return projector
    except Exception as e:
        print(f"Error in orthogonalizing projector")
        raise e

def _orthogonalize_projector(projector1: np.ndarray) -> np.ndarray:
    return projector1 @ invsqrt(dagger(projector1)@projector1)

# TODO: Make sure this reproduces the same result as building the rotation vectors from the inertia tensor for an overcomplete list of rotation vectors for a linear molecule
def safe_orthogonalize_projector(projector1):
    try:
        return orthogonalize_projector(projector1)
    except Exception as e:
        print(f"Error in regular orthogonalization of projector - attemping progressive orthogonalization")
        vectors = [v for v in projector1.T]
        vectors = remove_parallel_vectors_loop(vectors)
        # vectors = progressively_orthogonalize_vectors(vectors)
        return np.array(vectors).T

        


def progressively_orthogonalize_vectors(vectors: list[np.ndarray]) -> list[np.ndarray]:
    try:
        return _progressively_orthogonalize_vectors(vectors)
    except Exception as e:
        print(f"Error in progressively orthogonalizing vectors")
        raise e


def _progressively_orthogonalize_vectors(vectors: list[np.ndarray]) -> list[np.ndarray]:
    vector_array = np.array(vectors, dtype=np.float64)
    for j in range(len(vectors)):
        v1 = vector_array[j, :]/np.linalg.norm(vector_array[j, :])
        vector_array[j, :] = v1
        for k in range(j+1, len(vectors)):
            v2 = vector_array[k, :]/np.linalg.norm(vector_array[k, :])
            overlap = np.dot(v1, v2)
            orth_v2 = v2 - overlap * v1
            vector_array[k, :] = orth_v2 / np.linalg.norm(orth_v2)
    return list(vector_array)
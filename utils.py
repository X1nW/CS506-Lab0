import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    return the scalar dot product of the two vectors.
    '''
    return np.dot(v1, v2)

def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    '''
    return dot_product(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    '''
    best_similarity = -1  # Initialize with the lowest possible cosine similarity
    best_index = -1       # Index of the best match
    for i, vector in enumerate(vectors):
        similarity = cosine_similarity(target_vector, vector)
        if similarity > best_similarity:
            best_similarity = similarity
            best_index = i
    return best_index

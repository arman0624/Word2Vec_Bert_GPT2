import numpy as np
from scipy.stats import spearmanr


def get_similarity_scores(embeddings1, embeddings2):
    """
    Function to compute the similarity scores between two sets of embeddings.
    """
    similarity_scores = []
    for idx in range(len(embeddings1)):
        dot_product = np.dot(embeddings1[idx], embeddings2[idx])
        if isinstance(dot_product, np.ndarray):
            dot_product = dot_product[0]  # Extract scalar value
        similarity_scores.append(round(dot_product, 6))
    return similarity_scores


def compute_spearman_correlation(similarity_scores, human_scores):
    """
    Function to compute the Spearman correlation between the similarity scores and human scores (labels).
    """
    return round(spearmanr(similarity_scores, human_scores).correlation, 6)

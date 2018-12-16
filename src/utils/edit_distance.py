import numpy as np


def levenshtein_distance(seq1, seq2):
    """ Levenshtein edit-distance
    https://en.wikipedia.org/wiki/Levenshtein_distance """

    n = len(seq1)
    m = len(seq2)

    if n == 0 or m == 0:
        return max(n, m)

    # levenstein[i, j] represents string edit distance between seq1[:i] and seq2[:j]
    levenshtein = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(0, n + 1):
        for j in range(0, m + 1):
            if min(i, j) == 0:
                levenshtein[i, j] = max(i, j)
            else:
                levenshtein[i, j] = min(levenshtein[i - 1, j] + 1,
                                        levenshtein[i, j - 1] + 1,
                                        levenshtein[i - 1, j - 1] + (0 if seq1[i - 1] == seq2[j - 1] else 1))

    return levenshtein[n, m]
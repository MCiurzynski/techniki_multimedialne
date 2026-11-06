
import numpy as np

def bhattacharyya(hist, other_hist):
        numerator = np.sum(np.sqrt(hist * other_hist))
        
        denominator = np.sqrt(np.sum(hist) * np.sum(other_hist))
        
        if denominator == 0:
            return 1.0 

        val = numerator / denominator
        val = np.clip(val, 0.0, 1.0)
        
        value = 1.0 - np.sqrt(1.0 - val)
        return float(value)

def pearson_correlation(hist, other_hist):
    # Centralizing
    hist = hist - np.mean(hist)
    other_hist = other_hist - np.mean(other_hist)

    # Pearson correlation
    denominator = np.sqrt(np.sum(hist ** 2) * np.sum(other_hist ** 2))
    if denominator == 0:
        return 0.0

    value = np.sum(hist * other_hist) / denominator
    return float(value)

def chi_square(hist, other_hist):
    diff_sq = (hist - other_hist) ** 2
    # Dzielimy tylko tam, gdzie hist != 0, w przeciwnym razie wpisujemy 0
    return np.sum(np.where(hist != 0, diff_sq / hist, 0))
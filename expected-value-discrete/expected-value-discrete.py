import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x, dtype = float)
    p = np.array(p, dtype = float)
    
    if len(x) != len(p):
        raise ValueError("x is not equal to p")
    if np.any(p<0):
        raise ValueError("Probabilities are less than 0")
    if not np.isclose(np.sum(p),1.0):
        raise ValueError("Prbablities normalization not met")
        
    expected_value = np.sum(x * p)
    return expected_value

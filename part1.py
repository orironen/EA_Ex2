import cvxpy as cp
import numpy as np

def egalitarian_division(matrix):
    """
    Compute an egalitarian division for m agents and n resources.

    Example usage :

    >>> matrix = [
    ...     [81, 19, 1],
    ...     [70, 1, 29]
    ... ]
    >>> alloc = egalitarian_division(matrix)
    Agent #1 gets 0.53 of resource #1, 1.00 of resource #2, 0.00 of resource #3
    Agent #2 gets 0.47 of resource #1, 0.00 of resource #2, 1.00 of resource #3
    >>> alloc.shape
    (2, 3)
    """
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape
    resources = cp.Variable((m, n))
    # תועלות לכל אדם
    utilities = cp.sum(cp.multiply(matrix, resources), axis=1)
    # מינימום תועלת לכל אדם
    min_util = cp.Variable()

    # מגבלות
    constraints = [
        resources >= 0,
        resources <= 1,
        cp.sum(resources, axis=0) == 1,# כל משאב מחולק לחלוטין
        utilities >= min_util         
    ]

    prob = cp.Problem(cp.Maximize(min_util), constraints)
    prob.solve()

    alloc = resources.value

    for j in range(m):
        fracs = ", ".join(f"{alloc[j, i]:.2f} of resource #{i+1}" for i in range(n))
        print(f"Agent #{j+1} gets {fracs}")
    return alloc

if __name__ == "__main__":
    import doctest
    doctest.testmod()
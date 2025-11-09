import cvxpy as cp
import numpy as np
from itertools import combinations

def leximin_division(matrix):
    """
    Compute a leximin-egalitarian division for m agents and n resources.

    Example usage:

    >>> matrix = [
    ...     [81, 19, 1],
    ...     [70, 1, 29]
    ... ]
    >>> alloc = leximin_division(matrix) 
    Agent #1 gets 0.53 of resource #1, 1.00 of resource #2, 0.00 of resource #3
    Agent #2 gets 0.47 of resource #1, 0.00 of resource #2, 1.00 of resource #3
    >>> alloc.shape
    (2, 3)
    """
    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape
    resources = cp.Variable((m, n))
    utilities = cp.sum(cp.multiply(matrix, resources), axis=1)
    constraints = [
        resources >= 0,
        resources <= 1,
        cp.sum(resources, axis=0) == 1
    ]

    # רשימת משתנים לכל איטרציה
    values = []

    # איטרציות לפי מספר האנשים
    for k in range(1, m + 1):
        z = cp.Variable()  
        for prev in values:
            constraints.append(utilities >= prev)
        for combo in combinations(range(m), k):
            constraints.append(cp.sum(utilities[list(combo)]) >= z)
        # פתרון בעיה נוכחית
        prob = cp.Problem(cp.Maximize(z), constraints)
        prob.solve()
        values.append(z.value)

    alloc = resources.value
    alloc = np.where(np.abs(alloc) < 1e-10, 0.0, alloc)
    for j in range(m):
        fracs = ", ".join(f"{alloc[j, i]:.2f} of resource #{i+1}" for i in range(n))
        print(f"Agent #{j+1} gets {fracs}")
    return alloc

if __name__ == "__main__":
    import doctest
    doctest.testmod()

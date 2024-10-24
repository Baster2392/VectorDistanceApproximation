import numpy as np
from scipy.optimize import fsolve
import itertools
import csv


def equation(P, I, number_of_layers, q, C):
    # First
    input2first_layer = P * I
    last_layer2output = P * (q ** (number_of_layers - 1))
    # Calculation of the connections between hidden layers
    # is equivalent to the sum of the first n-1 terms of a geometric progression
    r = q ** 2
    n = number_of_layers - 1
    a = q * P ** 2
    hidden2hidden = a * (1 - r ** n) / (1 - r) if q != 1 else n * P ** 2
    return input2first_layer + hidden2hidden + last_layer2output - C


def find_P(I, n, q, C):
    # Initial guess for P
    P_guess = 1.0

    # Using fsolve to find the root
    P_solution, = fsolve(equation, P_guess, args=(I, n, q, C))

    return P_solution


I_list = [1000]  # input dimensions
n_list = [2]  # number of layers
q_list = [0.75]  # factor of convergence
C_list = [600000000]     # [40000000, 100000000, 200000000, 300000000, 400000000, 480000000, 520000000, 550000000, 580000000, 600000000]  # approximate complexity/cost

# Perform grid search
results = []

for I, n, q, C in itertools.product(I_list, n_list, q_list, C_list):
    P = find_P(I, n, q, C)
    Pq = P * q
    # Calculate and store additional terms
    Pq_terms = [round(P * (q ** i)) for i in range(n)]
    Pq_terms_1 = [P * (q ** i) for i in range(n)]
    P = round(P)
    results.append((I, n, q, C, P, Pq_terms, Pq_terms_1))

filtered_results = []

for result in results:
    I, n, q, C, P, Pq_terms, Pq_terms_1 = result
    term1 = P * I + P * (q ** (n - 1))
    term2 = 2 * (P ** 2) * q + (n - 2) * q ** 2
    term2 = 0.5 * term2 * (n - 1)
    res = term1 + term2
    Cp = 0
    Cz = 0
    for i in range(n):
        t1 = Pq_terms[i]
        t2 = Pq_terms[i - 1] if i > 0 else I
        c1 = Pq_terms_1[i]
        c2 = Pq_terms_1[i - 1] if i > 0 else I
        Cz += c1 * c2
        Cp += t1 * t2
    Cp += Pq_terms[n - 1]
    Cz += Pq_terms_1[n - 1]
    filtered_results.append((I, n, q, C, P, Pq_terms, Cp))

# Apply the first filter: if n=1 then leave only q=1
filtered_results = [res for res in filtered_results if not (res[1] == 1 and res[2] != 1)]

# Apply the second filter: if for the same C and n there are multiple results with the same P, leave only one that has Cp closest to C
final_results = []
for key, group in itertools.groupby(filtered_results, key=lambda x: (x[3], x[1], x[4])):
    group = list(group)
    closest_to_C = min(group, key=lambda x: abs(x[6] - x[3]))
    final_results.append(closest_to_C)

# Apply the third filter: filter out all results where abs(C - Cp)/C > 0.1
# final_results = [res for res in final_results if abs(res[3] - res[6]) / res[3] <= 0.1]
# probably obsolete since my math is now correct

# Apply the fourth filter: filter out any results where any of Pq_terms is 0
final_results = [res for res in final_results if all(term != 0 for term in res[5])]
# Apply the fifth filter: filter out any results where P <= 1000
# final_results = [res for res in final_results if res[4] <= 1000]
# Print final filtered results
for result in final_results:
    I, n, q, C, P, Pq_terms, Cp = result
    print(f"For I={I}, n={n}, q={q}, C={C}, P is {P}")
    print(f"  Pq terms: {[round(term, 4) for term in Pq_terms]}")
    print(f"  C: {Cp}")
print(f'Number of results: {len(final_results)}')

with open('results.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(
    #     ['Input dimension', 'Number of Layers', 'Factor q', 'Meant Complexity', 'Actual Complexity', 'First Layer',
    #      'Layers'])
    for result in final_results:
        I, n, q, C, P, Pq_terms, Cp = result
        writer.writerow([I, n, q, C, Cp, P, ','.join([str(round(term, 4)) for term in Pq_terms])])

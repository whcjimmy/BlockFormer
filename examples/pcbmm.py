import pdb
import time
from multiprocessing import Pool
import numpy as np

# Bert-Base
m = 128
n = 768
k = 64
S = 32768
d = 64

# Llama-3-8B
'''
m = 8
n = 4096
k = 128
S = 32768
d = 128
'''

def generate_random_matrix(rows, cols):
    return np.random.randint(0, 10, size=(rows, cols))

def generate_zero_matrix(rows, cols):
    return np.zeros((rows, cols))

def generate_ct_slots(num_slots):
    return np.zeros(num_slots)

# ORIGINAL
A = generate_random_matrix(m, n)
B = generate_random_matrix(n, k)
C = A @ B

# OPTIMIZED
# a_ct = generate_ct_slots(S)
# b_pt = generate_ct_slots(S)
O = generate_zero_matrix(m, k)

def worker(num):
    a_ct = generate_ct_slots(S)
    b_pt = generate_ct_slots(S)
    o_ct = generate_ct_slots(S)

    for i in range(n//d):
        # print(i)
        for b in range(S//(m*d)):
            begin = b * m * d
            end = (b + 1) * m * d
            a_ct[begin:end] = A[:, i*d:(i+1)*d].flatten()
        for b in range(m):
            b_pt[b * d:(b + 1) * d] = B[i*d:(i+1)*d, num]

        o_ct += a_ct * b_pt

    for i in range(int(np.log2(d)) - 1, -1, -1):
        t_ct = o_ct
        o_ct += np.roll(t_ct, -np.power(2, i))

    return o_ct

start = time.time()
with Pool(processes=64) as pool:
    results = pool.map(worker, range(k))
    for c in range(k):
        o_ct = results[c]
        for i in range(m):
            O[i, c] = o_ct[i * d]
end = time.time()

print("=== ORIGINAL ===")
print(C)
print("=== PROPOSED ===")
print(O)
print("Runtime of Proposed PCBMM: ", end - start)
print("EQUAL? ", np.array_equal(C, O))

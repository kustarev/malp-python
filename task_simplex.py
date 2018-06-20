import numpy as np
from malp3 import Task

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	dim = 100
	coeffs = [ float(1 * i + 1) for i in range(dim) ]
	t = Task(max_iter = 1000, step = 0.01, iter_start = 1, iter_exp = 3.0)
	for i in range(dim):
		t.new_variable()
		t.add_constraint([i],[1.0], 0.0)
		t.add_coeff(i, coeffs[i])
	t.add_constraint(list(range(dim)), [-1.0] * dim, 1.0)
	start = [ 1.0 / (1.0 + dim) ] * dim
	res = t.solve(start)
	print('Solution:')
	print(res)
	print('Verifying optimized versus expected solution..')
	atol = 0.001
	assert np.isclose(res[-1], 1.0, atol = atol)
	assert all(np.isclose(res[:-1], np.zeros(dim - 1), atol = atol))
	print('done')
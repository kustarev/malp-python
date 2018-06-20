import numpy as np
from malp3 import Task

if __name__ == '__main__':
	np.set_printoptions(suppress=True)
	np.random.seed(0)
	dim = 100
	gmv = 1.0
	target_num = 10
	max_abs_pos = gmv / target_num
	t = Task(max_iter = 1000, step = 1.0, iter_start = 1e+6, iter_exp = 3.0)
	ideal_pos = sorted(np.random.rand(dim))
	optim_start = []
	for i in range(dim):
		pos_idx = t.new_variable(group = "pos")
		optim_start.append(0)
		abs_pos_idx = t.new_variable(group = "abs_pos")
		optim_start.append(0.5 * gmv / dim)
		t.add_constraint([abs_pos_idx, pos_idx], [1, -1], 0)
		t.add_constraint([abs_pos_idx, pos_idx], [1, 1], 0)
		t.add_constraint([abs_pos_idx], [-1], max_abs_pos)
		t.add_coeff(pos_idx, ideal_pos[i])
	t.add_constraint(t.get_idx_group("abs_pos"), [-1.0] * dim, gmv)
	res = t.solve(optim_start)
	result_pos = res[t.get_idx_group("pos")]
	print('Verifying optimized versus expected solution..')
	atol = 0.001
	assert all(np.isclose(result_pos[-target_num:], max_abs_pos * np.ones(target_num), atol = atol))
	assert all(np.isclose(result_pos[:-target_num], np.zeros(dim - target_num), atol = atol))
	print('done')




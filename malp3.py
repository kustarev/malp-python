import numpy as np
from collections import defaultdict

class Task:
	def __init__(self, max_iter = 100, step = 1.0, verbose = True, iter_start = 0, iter_exp = 1.0):
		self.max_iter = max_iter
		self.step = step
		self.verbose = verbose
		self.bounds = []
		self.constraints = []
		self.cons_idx = []
		self.coeffs = []
		self.idx_groups = defaultdict(list)
		self.iter_start = iter_start
		self.iter_exp = iter_exp
	def _assert_idx(self, idx):
		assert 0 <= idx < len(self.coeffs)
	def new_variable(self, group = ""):
		self.coeffs.append(0)
		res_idx = len(self.coeffs) - 1
		self.idx_groups[group].append(res_idx)
		return res_idx
	def get_idx_group(self, group):
		return self.idx_groups[group]
	def add_constraint(self, idx_list, coeff_list, bound):
		for idx in idx_list:
			self._assert_idx(idx)
		assert len(idx_list) == len(coeff_list)
		self.bounds.append(bound)
		self.constraints.append(coeff_list)
		self.cons_idx.append(idx_list)
	def add_coeff(self, idx, val):
		self._assert_idx(idx)
		self.coeffs[idx] = val
	def _set_task(self):
		self.B = np.array(self.bounds)
		self.C = np.array(self.coeffs)
		self.A = np.zeros((len(self.bounds), len(self.coeffs)))
		for i, (idx_list, coeff_list) in enumerate(zip(self.cons_idx, self.constraints)):
			self.A[i, idx_list] = coeff_list
		self.pinvA = np.linalg.pinv(self.A)
		self.yC = np.dot(self.C, self.pinvA)
	def solve(self, start):
		self._set_task()
		x = np.array(start)
		y = self._y(x)
		iterations = 1
		while True:
			z = np.sqrt(y)
			new_y = y + self.step * self._gradient(z, iterations)
			assert self._feasible(new_y)
			if self.verbose:
				print('Iteration {}'.format(iterations))	
				print('Solution vector:')
				print(self._x(y))
			if iterations == self.max_iter:
				break
			iterations += 1
			y = new_y
		if self.verbose:
			print('Stopped after {} iteration(s)'.format(iterations))
		return self._x(y)
	def _eval(self, y):
		return np.dot(self.yC, y)
	def _y(self, x):
		return np.dot(self.A, x) + self.B
	def _x(self, y):
	 	return np.dot(self.pinvA, y - self.B)
	def _feasible(self, y):
		return np.all(y > 0) 
	def _gradient(self, z, iter):
		# compute gradient of the corresponding quadratic function in global Euclidean space
		zC = self.yC * z 
		# compute iteration decay multiplier (yields to 0 as iteration number increases)
		iter_decay = 1 / (self.iter_start + iter ** self.iter_exp)
		# update the gradient to stay away from boundaries
		zC += iter_decay / z 
		# get manifold tangent space in the current point
		tg = np.dot(self.A.T, np.diag(1 / z)) 
		# project the gradient onto the tangent space
		grad_coeffs = np.dot(zC, np.linalg.pinv(tg)) 
		grad_z = np.dot(tg.T, grad_coeffs) 
		# map the gradient back onto the polytope affine image
		grad_y = grad_z * z 
		return grad_y
		

import cplex
import argparse
import json
import os

import time


def solve(problem_path=None, solution_file=None, warm_start=None, abstol=None, relative_mip_gap=None):
	if problem_path is None:
		raise ValueError("must provide path to problem")

	if solution_file is None:
		solution_file = problem_path.replace('.lp', '.sol')

	cplex_log_file = problem_path.replace('.lp', '_cplex.log')

	cpx = cplex.Cplex()
	if abstol is not None:
		print(f'Setting abs mip tol to {abstol}')
		cpx.parameters.mip.tolerances.absmipgap.set(abstol)
	if relative_mip_gap:
		print(f'Setting relative mip gap to {relative_mip_gap}')
		cpx.parameters.mip.tolerances.mipgap.set(relative_mip_gap)
	start_time = time.time()
	try:
		with open(cplex_log_file, 'w') as cplexlog:
			cpx.set_results_stream(cplexlog)
			cpx.set_warning_stream(cplexlog)
			cpx.set_error_stream(cplexlog)
			cpx.set_log_stream(cplexlog)
			#time.sleep(5)
			cpx.read(problem_path)
			if warm_start is not None:
				if not os.path.exists(warm_start):
					raise ValueError(f"File {warm_start} should be a solution file. But does not exist?")
				cpx.start.read_start(warm_start)
			cpx.solve()
			status = cpx.solution.get_status()
			status_string = cpx.solution.get_status_string()
			obj = cpx.solution.get_objective_value()
			print(f"Status {status}: {status_string}")
			print(f"Objective {obj}")
			cpx.solution.write(solution_file)
			print("Wrote solution to %s" % solution_file)
			cpx.end()
	except Exception as e:
		print(e)
		status = '1217'
		status_string = 'Infeasible'
		print("Infeasible, no solution to write!")
		obj=-10000000

	end_time = time.time()

	with open(problem_path.replace('.lp', '_status.json'),'w') as f:
		status_dict = {'time': end_time-start_time,
						'status': f'{status}: {status_string}',
						'objective': obj}
		f.write(json.dumps(status_dict))

	return solution_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem_path", help="problem file to read", default='./data/problem.lp')
    parser.add_argument("--solution_file", help="solution file to save in CPLEX sol format", default=None)
    parser.add_argument("--warm_start", help="solution file to warm start", default=None)
    parser.add_argument("--results_file", help="place to store results of analysis", default=None)
    parser.add_argument("--abstol", help="CPLEX param abstol", default=None, type=int)
    args = parser.parse_args()
    solve(**vars(args))

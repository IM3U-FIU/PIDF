from int_alg import int_alg_mnist
from synth_gen import generate_data
import argparse
import time
from types import SimpleNamespace

args = SimpleNamespace(
    pix_num=123,      # Replace with your desired pixel number
    num_iters=10     # Replace with your desired number of iterations
)


'''parser = argparse.ArgumentParser()
parser.add_argument("pix_num", help="pix_num", type=int)
parser.add_argument("num_iters", help="Number of iterations", type=int)
args = parser.parse_args()'''

if __name__ == "__main__":
    name = 'mnist'
    num_iters = args.num_iters

    obs, acs = generate_data(nme=name)
    start_time = time.time()
    
    # Run the algorithm
    int_alg_mnist(obs, acs, num_iters, args.pix_num,  name).run()
    
    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    print(num_iters)
    print(f"Time taken to run int_alg: {time_taken:.2f} seconds")

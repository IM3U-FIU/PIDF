from int_alg import int_alg
from synth_gen import generate_data
from vizualize import visualize_pidf
from feature_selection import run_feature_selection
import argparse
import numpy as np
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("name", help="Dataset name", type=str)
parser.add_argument("num_iters", help="Number of iterations", type=int)
parser.add_argument("feature_selection", help="True to run feature selection, False to run int_alg and visualize", type=str)
args = parser.parse_args()

def alt_main(obs, acs, name="Custom", feature_selection=False,num_iters=200,scalable=False):
  if feature_selection.find('True')==-1:
    # Run the int_alg and visualize branch
    start_time = time.time()

    # Run the algorithm
    data, data_std, reds_n_syns = int_alg(obs, acs, num_iters, scalable=scalable).run()

    # End timing
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to run int_alg: {time_taken:.2f} seconds")

    # Save results to .npy files
    with open(f'interpretability_{args.name}.pickle', 'wb') as h:
      pickle.dump(data, h)
    with open(f'interpretability_std_{args.name}.pickle', 'wb') as h:
      pickle.dump(data_std, h)
    with open(f'syns_and_reds_{args.name}.pickle', 'wb') as h:
      pickle.dump(reds_n_syns, h)

    # Visualize results
    visualize_pidf(name)
  else:
    # Run feature selection on the provided dataset name with the specified number of iterations.
    start_time = time.time()
    run_feature_selection(names=[name], num_iters=num_iters)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to run feature selection: {time_taken:.2f} seconds")


if __name__ == "__main__":
    if args.feature_selection.find('True')==-1:
        # Run the int_alg and visualize branch
        obs, acs = generate_data(nme=args.name)
        start_time = time.time()

        # Run the algorithm
        data, data_std, reds_n_syns = int_alg(obs, acs, args.num_iters, scalable=False).run()

        # End timing
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken to run int_alg: {time_taken:.2f} seconds")

        # Save results to .pickle files
        with open(f'interpretability_{args.name}.pickle', 'wb') as h:
          pickle.dump(data, h)
        with open(f'interpretability_std_{args.name}.pickle', 'wb') as h:
          pickle.dump(data_std, h)
        with open(f'syns_and_reds_{args.name}.pickle', 'wb') as h:
          pickle.dump(reds_n_syns, h)

        # Visualize results
        visualize_pidf(args.name)
    else:
        # Run feature selection on the provided dataset name with the specified number of iterations.
        start_time = time.time()
        run_feature_selection(names=[args.name], num_iters=args.num_iters)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken to run feature selection: {time_taken:.2f} seconds")

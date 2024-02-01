import argparse
from algorithm.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--nmf', 
                    type=str, 
                    choices=["L2NormNMF", "KLDvergenceNMF", "ISDivergenceNMF", "L21NormNMF", 
                             "HSCostNMF", "L1NormRegularizedNMF", "CappedNormNMF", "CauchyNMF"], 
                    help='NMF Algorithm.', 
                    default='L1NormRegularizedNMF')

parser.add_argument('--dataset', 
                    choices=["ORL", "YaleB"],
                    type=str, help='Dataset.', 
                    default='YaleB')

parser.add_argument('--reduce', 
                    type=int, 
                    help='Reduce. Options: 1, 3', 
                    default=3)

parser.add_argument('--noise_type', 
                    type=str, 
                    choices=["uniform", "gaussian", "laplacian", "salt_and_pepper", "block"],
                    help='Noise type.', 
                    default='salt_and_pepper')

parser.add_argument('--noise_level', 
                    type=float, 
                    help='Noise level. Uniform, Gassian, Laplacian: [.1, .3], Salt and Pepper: [.02, .10], Block: [10, 20]', 
                    default=0.02)

parser.add_argument('--random_state', 
                    type=int, 
                    help='Random state. Options: 0, 42, 99, 512, 3407', 
                    default=99)

parser.add_argument('--scaler', 
                    choices=["MinMax", "Standard"],
                    type=str, 
                    help='Scaler.', 
                    default='MinMax')

parser.add_argument('--max_iter', 
                    type=int,
                    help='Max iteration', 
                    default=500)

parser.add_argument('--verbose', 
                    action='store_false',
                    help='Verbose')

parser.add_argument('--idx', 
                    type=int, 
                    help='Image index', 
                    default=9)

parser.add_argument('--imshow', 
                    action='store_false', 
                    help='Show image')

args = parser.parse_args()

pipeline = Pipeline(nmf=args.nmf, 
                    dataset=args.dataset,
                    reduce=args.reduce,
                    noise_type=args.noise_type,
                    noise_level=args.noise_level,
                    random_state=args.random_state,
                    scaler=args.scaler)

# Run the pipeline
pipeline.execute(max_iter=args.max_iter, verbose=args.verbose) # Parameters: max_iter: int, convergence_trend: bool, matrix_size: bool, verbose: bool
pipeline.evaluate(idx=args.idx, imshow=args.imshow) # Parameters: idx: int, imshow: bool
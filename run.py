from algorithm.pipeline import Pipeline

pipeline = Pipeline(nmf='L1NormRegularizedNMF', # Options: 'L2NormNMF', 'L1NormNMF', 'KLdivergenceNMF', 'ISdivergenceNMF', 'RobustNMF', 'HypersurfaceNMF', 'L1NormRegularizedNMF', 'CappedNormNMF', 'CauchyNMF'
                    dataset='YaleB', # Options: 'ORL', 'YaleB'
                    reduce=3, # ORL: 1, YaleB: 3
                    noise_type='salt_and_pepper', # Options: 'uniform', 'gaussian', 'laplacian', 'salt_and_pepper', 'block'
                    noise_level=0.02, # Uniform, Gassian, Laplacian: [.1, .3], Salt and Pepper: [.02, .10], Block: [10, 20]
                    random_state=99, # 0, 42, 99, 512, 3407 in our experiments
                    scaler='MinMax') # Options: 'MinMax', 'Standard'

# Run the pipeline
pipeline.run(max_iter=500) # Parameters: max_iter: int, convergence_trend: bool, matrix_size: bool, verbose: bool
pipeline.evaluate(idx=9, imshow=True) # Parameters: idx: int, imshow: bool
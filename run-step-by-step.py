import numpy as np

from algorithm.datasets import load_data
from algorithm.preprocess import NoiseAdder, MinMaxScaler
from algorithm.sample import random_sample
from algorithm.nmf import L1NormRegularizedNMF

DATA_DIR = 'data/CroppedYaleB'
REDUCE = 3
RANDOM_STATE = 99
SAMPLE_RATE = 0.9
NOISE_TYPE = 'salt_and_pepper'
NOISE_LEVEL = 0.02

hparams = {}

scaler = MinMaxScaler()
noise_adder = NoiseAdder(random_state=RANDOM_STATE)
nmf = L1NormRegularizedNMF()

X_hat, Y_hat = load_data(root=DATA_DIR, reduce=REDUCE)
X_hat, Y_hat = random_sample(X_hat, Y_hat, SAMPLE_RATE, random_state=RANDOM_STATE)

X_hat_scaled = scaler.fit_transform(X_hat)
add_noise_ = getattr(noise_adder, f'add_{NOISE_TYPE}_noise')
_, X_noise = add_noise_(X_hat, noise_level=NOISE_LEVEL)
X_noise_scaled = scaler.transform(X_noise)
X_noise_scaled += np.abs(np.min(X_noise_scaled)) * np.abs(np.min(X_noise_scaled)) * int(np.min(X_noise_scaled) < 0)

nmf.fit(X_noise_scaled, len(set(Y_hat)), random_state=RANDOM_STATE, **hparams)
D, R = nmf.D, nmf.R
print(nmf.evaluate(X_hat_scaled, Y_hat, random_state=RANDOM_STATE))
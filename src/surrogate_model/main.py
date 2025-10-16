from gpu_run import run_on_device
from train import train_dense_network
from plot_prediction import plot_random_prediction
#run_on_device(train_dense_network, "models/new_fourier_features.keras")
run_on_device(plot_random_prediction, "models/old_fourier_features.keras")


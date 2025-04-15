#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import numpy as np
import tensorflow as tf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from helper import (
    load_dataset, split_dataset, single_sample_experiment, 
    compute_flops_mse_inference, plot_flops_mse_inference,
    plot_channel_comparison, plot_mse_vs_snr
)

# Argument parser
parser = argparse.ArgumentParser(description='Run model evaluation script.')
parser.add_argument('--input_file', required=True, help='Path to the model configuration JSON file')
parser.add_argument('--num_inferences', type=int, default=100, help='Number of inferences for profiling')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use during prediction (applies only to non-TensorRT models)')
args = parser.parse_args()

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

# Load model configuration
logging.info('Loading model configuration from JSON file...')
with open(args.input_file, 'r') as json_file:
    models_dict = json.load(json_file)

# Load and split dataset
dataset_path = models_dict["dataset"]["dataset_folder"] + models_dict["dataset"]["dataset_name"]
logging.info('Loading dataset from HDF5 file...')
trainData, trainLabels, trainPractical, trainLinearInterpolation, otherLabels = load_dataset(dataset_path)
(trainData_train, trainLabels_train, trainPractical_train, trainLinearInterpolation_train, otherLabels_train,
 trainData_validation, trainLabels_validation, trainPractical_validation, trainLinearInterpolation_validation, otherLabels_validation,
 trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, otherLabels_test) = split_dataset(
     trainData, trainLabels, trainPractical, trainLinearInterpolation, otherLabels, val_test_size=0.3)

first_dimension = trainLabels_test.shape[0]

# Load models and compute predictions
results = {"general_results": {}}
loaded_models = {}

logging.info('Beginning inference and metric evaluation for each model...')
for model_name, model_info in models_dict["models"].items():
    model_path = model_info["model_folder"] + model_info["model_file_name"]
    if model_name not in loaded_models:
        loaded_models[model_name] = load_model(model_path)

    model = loaded_models[model_name]
    logging.info(f'Performing Inference for (original/non-optimized) model {model_name}')
    y_pred = model.predict(trainData_test, batch_size = args.batch_size)

    y_true_reshaped = trainLabels_test.reshape(first_dimension, -1)
    y_pred_reshaped = y_pred.reshape(first_dimension, -1)

    results["general_results"][model_name] = {
        "predictions": y_pred,
        "mse": mean_squared_error(y_true_reshaped, y_pred_reshaped),
        "mae": mean_absolute_error(y_true_reshaped, y_pred_reshaped),
        "r2": r2_score(y_true_reshaped, y_pred_reshaped)
    }

# Add traditional estimators
results["general_results"]["Practical"] = {
    "predictions": trainPractical_test,
    "mse": mean_squared_error(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1)),
    "mae": mean_absolute_error(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1)),
    "r2": r2_score(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1))
}
results["general_results"]["LS"] = {
    "predictions": trainLinearInterpolation_test,
    "mse": mean_squared_error(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1)),
    "mae": mean_absolute_error(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1)),
    "r2": r2_score(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1))
}

# Output model performance
logging.info('Calculating performance improvement of models...')
for model_name, metrics in results["general_results"].items():
    print(f"Model: {model_name} | MSE: {metrics['mse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics['r2']:.4f}")

# Run FLOPS and inference profiling
random_index = np.random.randint(0, trainData_train.shape[0])
test_input = np.expand_dims(trainData_train[random_index], axis=0)
logging.info('Computing FLOPs, MSE, and inference time...')
data_for_plot_dict = compute_flops_mse_inference(
    loaded_models, results, test_input, trainData_test, trainLabels_test,
    num_inferences=args.num_inferences, CNN_2D=True, json_filename="flops_mse_inf_results.json"
)

# Generate final plot
plot_flops_mse_inference(
    data_for_plot_dict, filename="flops_mse_inf_bar_plot.png",
    plot_mse=True, plot_inf_time=True,
    title_font_size=18, tick_font_size=14,
    legend_font_size=14, label_font_size=16,
    round_val=2
)

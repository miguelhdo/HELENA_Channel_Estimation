#!/usr/bin/env python
# coding: utf-8

# =======================
# IMPORTS AND DEPENDENCIES
# =======================
from datetime import datetime
import json
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Reshape,
                                     Dense, Multiply, Add, Input, Flatten, SpatialDropout2D)
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam, AdamW, Nadam, Lion, Ftrl, Adafactor, Adadelta

from helper import (single_sample_experiment, plot_bubble_tradeoff, plot_radar_nmse_flops_inftime,
                         plot_channel_comparison, plot_flops_mse_inference, plot_flops_nmse_inference,
                         compute_mse_nmse_nmse_db, compute_flops_mse_inference, get_inference_optimized_models,
                         plot_mse_vs_snr, plot_flops_and_mse, load_dataset, split_dataset)

# =======================
# TENSORFLOW GPU CONFIGURATION
# =======================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')


def main():
    # =======================
    # ARGUMENT PARSING
    # =======================
    parser = argparse.ArgumentParser(description="Run CE model evaluation")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON config file')
    parser.add_argument('--prefix_output_files', type=str, default='output', help='Prefix for output filenames')
    parser.add_argument('--num_inferences', type=int, default=100, help='Number of inferences for FLOPs estimation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size to use during prediction (applies only to non-TensorRT models)')
    
    args = parser.parse_args()
    
    input_file = args.input_file
    prefix_output_files = args.prefix_output_files
    num_inferences = args.num_inferences
    batch_size = args.batch_size

    # =======================
    # LOAD MODEL CONFIGURATION
    # =======================
    with open(input_file, 'r') as json_file:
        models_dict = json.load(json_file)
        
    # =======================
    # SET DATASET PATHS AND PARAMETERS
    # =======================
    dataset_folder = models_dict["dataset"]["dataset_folder"]
    dataset_name = models_dict["dataset"]["dataset_name"]
    
    # Load the dataset (HDF5 format)
    dataset_path = dataset_folder + dataset_name
    
    trainData, trainLabels, trainPractical, trainLinearInterpolation, trainLeastSquares, otherLabels = load_dataset(dataset_path)
    
    print([trainData.shape, trainLabels.shape, trainPractical.shape,trainLinearInterpolation.shape,  trainLeastSquares.shape, otherLabels.shape])
    
    # Example of how to call the function:
    (trainData_train, trainLabels_train, trainPractical_train, trainLinearInterpolation_train, trainLeastSquares_train, otherLabels_train, trainData_validation, trainLabels_validation, trainPractical_validation, trainLinearInterpolation_validation, trainLeastSquares_validation, otherLabels_validation, trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, trainLeastSquares_test, otherLabels_test) = split_dataset(
        trainData=trainData,
        trainLabels=trainLabels,
        trainPractical=trainPractical,
        trainLinearInterpolation=trainLinearInterpolation,
        trainLeastSquares=trainLeastSquares,
        otherLabels=otherLabels,
        val_test_size=0.3
    )
    
    
    #Extra inputs for CE-ViT (SNRdB, delaySpread, dopplerShift)
    trainSNR_train = otherLabels_train[:,0]
    trainDoppler_train = otherLabels_train[:,2]
    trainDelay_train = otherLabels_train[:,3]
    
    trainSNR_validation = otherLabels_validation[:,0]
    trainDoppler_validation = otherLabels_validation[:,2]
    trainDelay_validation = otherLabels_validation[:,3]
    
    trainSNR_test = otherLabels_test[:,0]
    trainDoppler_test = otherLabels_test[:,2]
    trainDelay_test = otherLabels_test[:,3]
    
    
    # Print shapes to verify the splits
    print("Training Data shape:", trainData_train.shape)
    print("Validation Data shape:", trainData_validation.shape)
    print("Test Data shape:", trainData_test.shape)
    
    first_dimension = trainLabels_test.shape[0]
    
    
    # Dictionary to store prediction results and loaded models
    results = {}
    results["general_results"]={}
    loaded_models = {}
    
    # Assuming trainData_test, trainLabels_test, and first_dimension are predefined
    for model_name, model_info in models_dict["models"].items():
        # Get model folder and file name from the JSON
        model_folder = model_info["model_folder"]
        model_file_name = model_info["model_file_name"]
        print('Model name is ', model_file_name) 
        
        # Load the model if not already loaded and store it in loaded_models dictionary
        model_path = model_folder + model_file_name
        if model_name not in loaded_models:
            loaded_models[model_name] = load_model(model_path)
    
        # Get the loaded model from the dictionary
        model = loaded_models[model_name]
        model.summary()
    
        # Generate predictions
        if model_name in ['HELENA', 'LSiDNN 48', 'LSiDNN 1024', 'HELENA MHSA']:
            y_pred = model.predict(trainLeastSquares_test, batch_size=batch_size)
        elif model_name == "CE-ViT":
            y_pred = model.predict(x=[trainData_test, trainSNR_test, trainDoppler_test, trainDelay_test], batch_size=batch_size)
        else:
            y_pred = model.predict(trainData_test, batch_size=batch_size)
        
        # Reshape labels and predictions
        y_true_reshaped = trainLabels_test.reshape(first_dimension, -1)
        y_pred_reshaped = y_pred.reshape(first_dimension, -1)
    
        # Calculate metrics
        #mse = mean_squared_error(y_true_reshaped, y_pred_reshaped)
        mse = compute_mse_nmse_nmse_db(y_true_reshaped, y_pred_reshaped)[2]
        mae = mean_absolute_error(y_true_reshaped, y_pred_reshaped)
        r2 = r2_score(y_true_reshaped, y_pred_reshaped)
        
        # Store predictions and metrics in the results dictionary
        results["general_results"][model_name] = {
            "predictions": y_pred,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }
    
    
    # Add Practical CE results
    #practical_mse = mean_squared_error(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1))
    practical_mse = compute_mse_nmse_nmse_db(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1))[2]
    practical_mae = mean_absolute_error(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1))
    practical_r2 = r2_score(trainLabels_test.reshape(first_dimension,-1), trainPractical_test.reshape(first_dimension,-1))
    
    results["general_results"]["Practical"] = {
        "predictions": trainPractical_test,
        "mse": practical_mse,
        "mae": practical_mae,
        "r2": practical_r2
    }
    
    # Add Ls+LI CE results
    #mse_linear = mean_squared_error(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1))
    mse_linear = compute_mse_nmse_nmse_db(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1))[2]
    mae_linear = mean_absolute_error(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1))
    r2_linear = r2_score(trainLabels_test.reshape(first_dimension,-1), trainLinearInterpolation_test.reshape(first_dimension,-1))
    
    results["general_results"]["LS"] = {
        "predictions": trainLinearInterpolation_test,
        "mse": mse_linear,
        "mae": mae_linear,
        "r2": r2_linear
    }
    
    # Loop through the results dictionary and print the metrics for each model
    for model_name, metrics in results["general_results"].items():
        print(f"Model: {model_name}")
        print("MSE: %.4f" % metrics["mse"])
        print("MAE: %.4f" % metrics["mae"])
        print("R²: %.4f" % metrics["r2"])
        print("-" * 30)  # Separator for better readability
    
    
    # Find the model with the lowest MSE
    best_model_name = min(results["general_results"], key=lambda x: results["general_results"][x]["mse"])
    best_mse = results["general_results"][best_model_name]["mse"]
    
    # Print the best model and its MSE
    print(f"Best Model: {best_model_name}")
    print(f"Best MSE: {best_mse:.4f}")
    print("-" * 30)
    
    # Calculate and print percentage improvement with respect to other models
    for model_name, metrics in results["general_results"].items():
        if model_name != best_model_name:
            mse = metrics["mse"]
            improvement = ((mse - best_mse) / mse) * 100
            print(f"Model: {model_name}")
            print(f"MSE: {mse:.4f}")
            print(f"Improvement over {model_name}: {improvement:.2f}%")
            print("-" * 30)
    
    # Example: Use the function with the necessary variables
    random_index = 111  # Example index
    #random_index = 765, 134  # Example index
    cmax, cmin, results = single_sample_experiment(
        random_index=random_index,
        trainData_test=trainData_test,
        trainLabels_test=trainLabels_test,
        trainPractical_test=trainPractical_test,
        trainLinearInterpolation_test=trainLinearInterpolation_test,
        trainLeastSquares_test=trainLeastSquares_test,
        trainSNR_test=trainSNR_test, 
        trainDoppler_test=trainDoppler_test, 
        trainDelay_test=trainDoppler_test,
        loaded_models=loaded_models,
        results=results
    )
    
    # Print cmax and cmin
    print(f"cmax: {cmax}")
    print(f"cmin: {cmin}")
    
    # Print the MSE for each model, practical, and linear interpolation
    for key, mse_value in results["single_sample_experiment"][random_index]["mse"].items():
        print(f'MSE {key}: {mse_value}')
    
    # Initialize data_list and titles dynamically from results dictionary
    data_list = []
    titles = []
    
    # Retrieve the magnitudes for practical and linear interpolation
    practical_magnitude = results["single_sample_experiment"][random_index]["magnitudes"]["LS"]
    linearInterpol_magnitude = results["single_sample_experiment"][random_index]["magnitudes"]["LI"]
    
    # Append practical and linear interpolation magnitudes to data_list
    data_list.append(linearInterpol_magnitude)
    titles.append('LI')
    
    data_list.append(practical_magnitude)
    titles.append('LS')
    
    # Loop through each model's prediction magnitudes and add to data_list
    for model_name in loaded_models.keys():
        predicted_magnitude = results["single_sample_experiment"][random_index]["magnitudes"][model_name]
        data_list.append(predicted_magnitude)
        titles.append(f'{model_name}')
    
    # Append the actual label (channel) magnitude
    label_magnitude = results["single_sample_experiment"][random_index]["magnitudes"]["actual_channel"]
    data_list.append(label_magnitude)
    titles.append('Actual Channel')
    
    # Compute mse_list for each model and add None for the actual label
    mse_list = []
    mse_list.append(results["single_sample_experiment"][random_index]["mse"]["LI"])  # Linear CE MSE
    mse_list.append(results["single_sample_experiment"][random_index]["mse"]["LS"])  # Practical CE MSE
    
    # Loop through each model and append MSE values
    for model_name in loaded_models.keys():
        mse_value = results["single_sample_experiment"][random_index]["mse"][model_name]
        mse_list.append(mse_value)
    
    # Append None for the actual channel (no MSE for the true label)
    mse_list.append(None)
    
    #prefix_output_files = "V100"
    
    
    # =======================
    # PLOTTING AND VISUALIZATION UTILITIES
    # =======================
    plot_channel_comparison(data_list, titles, mse_list, cmap="rainbow", num_cols=5, vmin=cmin, vmax=cmax, filename=f"{prefix_output_files}_single_sample_ce_comparison.png")
    
    plot_mse_vs_snr(results, otherLabels_test, trainLabels_test, figsize=(10,8), filename=f"{prefix_output_files}_nmse_vs_snr.png")
    
    data_for_plot_dict = compute_flops_mse_inference(loaded_models, results,  trainData_test, trainLeastSquares_test, [trainSNR_test, trainDoppler_test, trainDelay_test], trainLabels_test, num_inferences=num_inferences, CNN_2D=True, json_filename=f"{prefix_output_files}_flops_nmse_inf_data.json")
    
    plot_radar_nmse_flops_inftime(data_for_plot_dict, filename=f"{prefix_output_files}_radar_nmse_flops_inf_plot.png")
    
    plot_bubble_tradeoff(data_for_plot_dict, filename=f"{prefix_output_files}_bubble_tradeoff_plot.png")
    
if __name__ == "__main__":
    main()
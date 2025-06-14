# Standard Library
import os
import time
import json

# Third-Party Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ONNX & TensorRT
import tf2onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import h5py
import numpy as np

def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        # List all groups
        keys = list(f.keys())
    
        # Access the specific dataset containing object references
        obj_ref_dataset = f[keys[1]]
    
        # Iterate through each object reference in the dataset
        for i, obj_ref in enumerate(obj_ref_dataset):
    
            # Dereference the object reference to access the actual object
            obj = f[obj_ref[0]]
            
            if i == 0:
                trainData = np.array(obj, dtype=obj.dtype)
                trainData = np.transpose(trainData, (3, 2, 1, 0))
            elif i == 1:
                trainLabels = np.array(obj, dtype=obj.dtype)
                trainLabels = np.transpose(trainLabels, (3, 2, 1, 0))
            elif i == 2:
                trainPractical = np.array(obj, dtype=obj.dtype)
                trainPractical = np.transpose(trainPractical, (3, 2, 1, 0))
            elif i==3:
                trainLinearInterpolation = np.array(obj, dtype=obj.dtype)
                trainLinearInterpolation = np.transpose(trainLinearInterpolation, (3, 2, 1, 0))
            elif i == 4:
                trainLS = np.array(obj, dtype=obj.dtype)
                trainLS = np.transpose(trainLS, (3, 2, 1, 0))
            else:
                #print(f"Object {i}: name={obj.name}, type={type(obj)}, shape={getattr(obj, 'shape', 'N/A')}")
                otherLabels = np.array(obj, dtype=obj.dtype)
                otherLabels = np.transpose(otherLabels, (1, 0))
                #SNRdB, profileIdx, delaySpread, dopplerShift
    return trainData, trainLabels, trainPractical, trainLinearInterpolation, trainLS, otherLabels

def split_dataset(trainData, trainLabels, trainPractical, trainLinearInterpolation, trainLeastSquares, otherLabels, val_test_size=0.3, random_state=42):
    # Number of samples
    n_samples = trainData.shape[0]

    # Generate indices for the dataset
    indices = np.arange(n_samples)

    # Split indices for training and the remaining (validation + test)
    print("Obtaining indexes for first split: Training and Validation + Test")
    train_indices, temp_indices = train_test_split(indices, test_size=val_test_size, random_state=random_state)

    print("Obtaining indexes for second split: Validation and Test")
    # Further split the remaining indices into validation and test
    validation_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=random_state)

    # Split the datasets using the indices
    trainData_train = trainData[train_indices]
    trainLabels_train = trainLabels[train_indices]
    trainPractical_train = trainPractical[train_indices]
    trainLinearInterpolation_train = trainLinearInterpolation[train_indices]
    trainLeastSquares_train = trainLeastSquares[train_indices]
    otherLabels_train = otherLabels[train_indices]

    trainData_validation = trainData[validation_indices]
    trainLabels_validation = trainLabels[validation_indices]
    trainPractical_validation = trainPractical[validation_indices]
    trainLinearInterpolation_validation = trainLinearInterpolation[validation_indices]
    trainLeastSquares_validation = trainLeastSquares[validation_indices]
    otherLabels_validation = otherLabels[validation_indices]

    trainData_test = trainData[test_indices]
    trainLabels_test = trainLabels[test_indices]
    trainPractical_test = trainPractical[test_indices]
    trainLinearInterpolation_test = trainLinearInterpolation[test_indices]
    trainLeastSquares_test = trainLeastSquares[test_indices]
    otherLabels_test = otherLabels[test_indices]
    print("Training, Validation, and Test split done!")

    return (trainData_train, trainLabels_train, trainPractical_train, trainLinearInterpolation_train, trainLeastSquares_train, otherLabels_train,
            trainData_validation, trainLabels_validation, trainPractical_validation, trainLinearInterpolation_validation, trainLeastSquares_validation, otherLabels_validation,
            trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, trainLeastSquares_test, otherLabels_test)


def plot_mse_vs_snr(results, otherLabels_test, trainLabels_test, filename="MSE_vs_SNR.png",
                    title_font_size=18, tick_font_size=14, label_font_size=16, legend_font_size=12, figsize=(10, 6)):
    # Unique SNR values
    unique_snr_values, unique_snr_counts = np.unique(otherLabels_test[:, 0], return_counts=True)

    # Initialize the mse_results dictionary dynamically
    mse_results = {
        "SNR": [],
        "MSE_practical": [],
        "MSE_linear_interpolation": []
    }

    # Add model names dynamically from the results dictionary
    #model_names = [model for model in results["general_results"].keys() if model not in ["LI", "LS"]]
    model_names = [model for model in results["general_results"].keys()]
    for model_name in model_names:
        mse_results[f"MSE_{model_name}"] = []

    # Loop through each SNR value and calculate the MSE for each model
    for snr, idx in zip(unique_snr_values, unique_snr_counts):
        indices = otherLabels_test[:, 0] == snr
        first_dimension = idx

        # Store results for SNR, practical, and linear interpolation
        mse_results["SNR"].append(snr)

        # Dynamically calculate MSE for each model in results
        for model_name in model_names:
            mse_model = compute_mse_nmse_nmse_db(
                trainLabels_test[indices].reshape(first_dimension, -1),
                results["general_results"][model_name]["predictions"][indices].reshape(first_dimension, -1)
            )[2]
            mse_results[f"MSE_{model_name}"].append(mse_model)

    # Plotting the main figure
    plt.figure(figsize=figsize)

    # Dynamically plot all models from the results
    markers = ['o', '>', 'v', 'p', 'd', 's', 'x', '+', '<', '*']  # Different markers for dynamic models
    colors = ['red', 'purple', 'orange', 'brown', 'cyan', 'blue', 'green', 'fuchsia', 'black', 'greenyellow', 'crimson', 'chocolate']  # Different colors for dynamic models
    linestyles = [':', '-', '--', '-.', '-', '--', '-.']  # Different line styles for dynamic models

    for idx, model_name in enumerate(model_names):
        plt.plot(mse_results["SNR"], mse_results[f"MSE_{model_name}"],
                 label=f"{model_name}", marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)])

    # Set labels and title with configurable font sizes
    plt.xlabel("SNR (dB)", fontsize=label_font_size)
    plt.ylabel("NMSE (dB)", fontsize=label_font_size)
    #plt.yscale('log')
    #plt.legend(fontsize=legend_font_size)
    #plt.legend(loc='upper right', fontsize=legend_font_size, ncol=(len(model_names) + 1) // 5, title="Models", bbox_to_anchor=(1.01, 0.98), title_fontsize=legend_font_size)
    #plt.legend(fontsize=legend_font_size)
    plt.legend(loc='upper right', fontsize=legend_font_size, ncol=(len(model_names) + 1) // 5, title="Estimators", title_fontsize=legend_font_size)
 
    plt.grid(True)

    # Set tick parameters for larger x and y tick labels
    plt.xticks(fontsize=tick_font_size)
    plt.yticks(fontsize=tick_font_size)

    # Inset axes for zooming in the range 15 < SNR <= 20 and 0.05 < y < 0.2
    ax_inset = inset_axes(plt.gca(), width="20%", height="20%", loc='lower left', borderpad=2)

    for idx, model_name in enumerate(model_names):
         ax_inset.plot(mse_results["SNR"], mse_results[f"MSE_{model_name}"],
                      marker=markers[idx % len(markers)], color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)])

    # Setting the limits for the zoomed-in area
    ax_inset.set_xlim(17.5, 20)
    ax_inset.set_ylim(-25, -20)  # Adjusted y-limits
    #ax_inset.set_yscale('log')
    ax_inset.grid(True)

    # Remove axis labels in the inset
    ax_inset.tick_params(axis='y', which='both', labelleft=False)

    # Set tick parameters for inset tick labels
    ax_inset.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Save the figure as a PNG image
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')

    plt.show()

def get_flops(model):
    input_specs = [
        tf.TensorSpec([1] + list(inp.shape[1:]), inp.dtype)
        for inp in model.inputs
    ]
    concrete = tf.function(lambda *inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(*input_specs)

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()

    # Calculate FLOPs using tf.profiler
    with tf.Graph().as_default():
        tf.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            tf.compat.v1.get_default_graph(),
            run_meta=run_meta,
            cmd='op',
            options=opts
        )
    return flops.total_float_ops if flops is not None else 0

def plot_flops_and_mse(models_dict, dummy_input, results, filename="flops_mse_plot.png", title_font_size=18, tick_font_size=14, legend_font_size=14, label_font_size=16):
    flops_dict = {}

    # Loop through each model in models_dict and calculate FLOPs
    for model_name, model in models_dict.items():

        if model_name == "CE-ViT":
            test_input =[
                np.expand_dims(test_data[random_index], axis=0),         # (1, 612, 14, 2)
                np.expand_dims(test_data_CE_ViT[0][random_index], axis=0),        # (1, 1)
                np.expand_dims(test_data_CE_ViT[1][random_index], axis=0),    # (1, 1)
                np.expand_dims(test_data_CE_ViT[2][random_index], axis=0)       # (1, 1)
            ]
            
            # Predict
            model(test_input)
        else:
            model(test_input)
            
        model(dummy_input)  # Forward pass to build the model
        flops = get_flops(model)
        flops_dict[model_name] = flops
        print(f'Total FLOPS for {model_name}: {flops}')

    # Get the model names dynamically from the dictionaries
    model_names = list(models_dict.keys())

    # Dynamically calculate FLOPs in billions and MSE values
    flops_values = [flops_dict[model] / 10.0**9 for model in model_names]
    mse_values = [round(results["general_results"][model]['mse'] / 10.0**(-3),3) for model in model_names]

    # Create a grouped bar plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar width
    bar_width = 0.35
    index = range(len(model_names))

    # Plot FLOPs with hatching patterns
    bars1 = ax1.bar(index, flops_values, bar_width, label='FLOPS', color='yellow', edgecolor='black', hatch='xx')

    # Plot MSE with different hatching patterns on the same axes
    ax2 = ax1.twinx()  # Create a second y-axis
    bars2 = ax2.bar([i + bar_width for i in index], mse_values, bar_width, label='MSE', color='orange', edgecolor='black', hatch='//')

    # Add values on top of the bars with increased font size and bold text
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                 ha='center', va='bottom', fontsize=tick_font_size,rotation=45)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2),
                 ha='center', va='bottom', fontsize=tick_font_size,rotation=45)

    # Labels and titles with configurable font sizes
    ax1.set_xlabel('Channel Estimation Model', fontsize=label_font_size)
    ax1.set_ylabel('FLOPS ($10^9$)', fontsize=label_font_size)
    ax2.set_ylabel('MSE ($10^{-3}$)', fontsize=label_font_size)

    # X-axis labels and ticks with configurable font size
    ax1.set_xticks([i + bar_width / 2 for i in index])
    ax1.set_xticklabels(model_names, fontsize=tick_font_size, rotation=45, ha='right')
    # X-axis labels and ticks with configurable font size
    ax1.set_xticks([i + bar_width / 2 for i in index])


    # Set tick parameters for both axes
    ax1.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Add a legend with a configurable font size
    legend_font = {'size': legend_font_size}
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, prop=legend_font)

    # Save the figure as a PNG image
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_channel_comparison(data_list, titles, mse_list=None, cmap="jet", font_size=18, tick_font_size=14, aspect_ratio=0.05, num_cols=4, hspace=0.4, wspace=0.3, vmin=0.0, vmax=1.0, filename="estimation_comparison.png"):
    """
    Plots a grid of images with corresponding titles and optional MSE values.

    Parameters:
        data_list (list of np.ndarray): List of data arrays to plot.
        titles (list of str): List of titles for each subplot.
        mse_list (list of float, optional): List of MSE values to include in the titles. Default is None.
        cmap (str, optional): Colormap to use for the images. Default is "jet".
        font_size (int, optional): Font size for the titles. Default is 18.
        tick_font_size (int, optional): Font size for the x and y tick labels. Default is 14.
        aspect_ratio (float, optional): Aspect ratio for the images. Default is 0.05.
        hspace (float, optional): Height space between rows. Default is 0.4.
        wspace (float, optional): Width space between columns. Default is 0.3.
        vmin (float, optional): Minimum value for colormap scaling. Default is 0.0.
        vmax (float, optional): Maximum value for colormap scaling. Default is 1.0.
    """

    num_plots = len(data_list)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows required

    # Create subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4 * num_rows), sharey=True)

    # Adjust the space between plots
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Keep track of the last image for colorbar reference
    im = None

    for i, data in enumerate(data_list):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]

        # Plot the data
        im = ax.imshow(np.abs(data), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect(aspect_ratio)

        # Set title with optional MSE
        if mse_list and mse_list[i] is not None:
            ax.set_title(f'{titles[i]}\nMSE: {mse_list[i]:.4f}', fontsize=font_size)
        else:
            ax.set_title(titles[i], fontsize=font_size)

        # Set tick parameters for larger x and y labels
        ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Hide any unused subplots
    for j in range(i + 1, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        ax = axs[row, col] if num_rows > 1 else axs[col]
        ax.axis('off')

    # Add color bar to the right of the subplots, using the last image for reference
    if im:
        fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)

    # Save the figure as a PNG image
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')

    plt.show()

def single_sample_experiment(random_index, trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, trainLeastSquares_test, trainSNR_test, trainDoppler_test, trainDelay_test, loaded_models, results, verbose = 0):
    """
    Perform a single sample experiment, generating predictions, calculating magnitudes and MSEs, and computing cmax and cmin.

    Parameters:
        random_index (int): Index of the sample to use.
        trainData_test (np.ndarray): Test data.
        trainLabels_test (np.ndarray): Ground truth labels for test data.
        trainPractical_test (np.ndarray): Practical test data.
        trainLinearInterpolation_test (np.ndarray): Linear interpolation data.
        loaded_models (dict): Dictionary of pre-loaded models.
        results (dict): Dictionary to store experiment results.

    Returns:
        cmax (float): Maximum magnitude value.
        cmin (float): Minimum magnitude value.
        results (dict): Updated results dictionary containing predictions, magnitudes, and MSEs.
    """
    # Retrieve the input, label, and other samples
    input_sample = trainData_test[random_index]
    input_sample_LS = trainLeastSquares_test[random_index]
    label_sample = trainLabels_test[random_index]
    practical_sample = trainPractical_test[random_index]
    linearInterpol_sample = trainLinearInterpolation_test[random_index]

    # Initialize or update results for the single sample experiment
    results["single_sample_experiment"] = results.get("single_sample_experiment", {})
    results["single_sample_experiment"][random_index] = {
        "predictions": {},
        "magnitudes": {},
        "mse": {}
    }

    # Loop through each model to calculate predictions, magnitudes, and MSE
    for model_name, model in loaded_models.items():
        # Predict using the model
        
        if model_name in ['HELENA', 'LSiDNN 48', 'LSiDNN 1024', 'HELENA MHSA']:
            predicted_sample = model.predict(np.expand_dims(input_sample_LS, axis=0))[0]
        elif model_name == "CE-ViT":
            # Pick one sample and wrap everything in a batch dimension
            input_sample = [
                np.expand_dims(trainData_test[random_index], axis=0),         # (1, 612, 14, 2)
                np.expand_dims(trainSNR_test[random_index], axis=0),        # (1, 1)
                np.expand_dims(trainDoppler_test[random_index], axis=0),    # (1, 1)
                np.expand_dims(trainDelay_test[random_index], axis=0)       # (1, 1)
            ]
            
            # Predict
            predicted_sample = model.predict(input_sample)[0]  # remove batch dim for result
        else:
            predicted_sample = model.predict(np.expand_dims(input_sample, axis=0))[0]
            
        # Store the prediction
        results["single_sample_experiment"][random_index]["predictions"][model_name] = predicted_sample

        # Compute and store the magnitude
        results["single_sample_experiment"][random_index]["magnitudes"][model_name] = np.linalg.norm(predicted_sample, axis=-1)

        # Compute and store the MSE between the label and the predicted sample
        #mse = np.mean((label_sample - predicted_sample) ** 2)
        mse = compute_mse_nmse_nmse_db(predicted_sample, label_sample)[2]
        results["single_sample_experiment"][random_index]["mse"][model_name] = mse

    # Add practical and linear interpolation results
    results["single_sample_experiment"][random_index]["predictions"]["LS"] = practical_sample
    results["single_sample_experiment"][random_index]["predictions"]["LI"] = linearInterpol_sample
    results["single_sample_experiment"][random_index]["predictions"]["actual_channel"] = label_sample

    # Compute and store the magnitudes for practical and linear interpolation
    results["single_sample_experiment"][random_index]["magnitudes"]["LS"] = np.linalg.norm(practical_sample, axis=-1)
    results["single_sample_experiment"][random_index]["magnitudes"]["LI"] = np.linalg.norm(linearInterpol_sample, axis=-1)
    results["single_sample_experiment"][random_index]["magnitudes"]["actual_channel"] = np.linalg.norm(label_sample, axis=-1)

    # Compute and store the MSE for practical and linear interpolation
    results["single_sample_experiment"][random_index]["mse"]["LS"] = compute_mse_nmse_nmse_db(label_sample,practical_sample)[2]
    results["single_sample_experiment"][random_index]["mse"]["LI"] = compute_mse_nmse_nmse_db(label_sample,linearInterpol_sample)[2]

    # Compute cmax and cmin using the magnitudes for each model, practical, linear, and label
    all_magnitudes = [
        results["single_sample_experiment"][random_index]["magnitudes"]["LS"],
        results["single_sample_experiment"][random_index]["magnitudes"]["LI"],
        results["single_sample_experiment"][random_index]["magnitudes"]["actual_channel"]
    ]

    # Loop through the models to add magnitudes to all_magnitudes list
    for model_name in loaded_models.keys():
        all_magnitudes.append(results["single_sample_experiment"][random_index]["magnitudes"][model_name])

    # Compute cmax and cmin
    cmax = np.max(np.abs(all_magnitudes))
    cmin = np.min(np.abs(all_magnitudes))

    return cmax, cmin, results

import onnx
from onnx import numpy_helper
import numpy as np

def convert_int64_to_int32(onnx_model_path, output_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Iterate through all initializers (weights/constants) in the graph
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.INT64:
            print(f"Converting {initializer.name} from INT64 to INT32")
            int64_data = numpy_helper.to_array(initializer)
            int32_data = int64_data.astype(np.int32)  # Convert to INT32
            initializer.CopyFrom(numpy_helper.from_array(int32_data, initializer.name))

    # Iterate through all node attributes to check for INT64 tensors
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                if any(isinstance(i, np.int64) for i in attr.ints):
                    print(f"Converting attribute {attr.name} in node {node.name} from INT64 to INT32")
                    attr.ints[:] = [int(i) for i in attr.ints]  # Convert list of int64 to int32

    # Save the modified ONNX model
    onnx.save(model, output_model_path)
    print(f"Saved modified ONNX model to {output_model_path}")


def mean_square_error_and_inf_time(
    dataset_input, dataset_label,
    *args,
    h_output, d_output, stream, context,
    CNN_2D=False, break_sample=0, print_interval=50, warming_samples=50, print_processing_sample=False
):
    """
    Calculates MSE and timing statistics.
    Supports single input or list-of-4 Ce-ViT inputs.

    Inputs:
    - dataset_input: single tensor or list of 4 tensors
    - dataset_label: ground truth
    - args: either h_input, d_input (for standard) or h1,h2,h3,h4, d1,d2,d3,d4 (for CE-ViT)
    """
    is_ce_vit = isinstance(dataset_input, list) and len(dataset_input) == 4

    if is_ce_vit:
        h_inputs = args[:4]
        d_inputs = args[4:8]
        input_shape = dataset_input[0].shape
        D = input_shape[0]
        if input_shape != dataset_label.shape:
            raise ValueError("Input and label shapes must match for CE-ViT")
    else:
        h_input, d_input = args[0], args[1]
        input_shape = dataset_input.shape
        D = input_shape[0]
        if input_shape != dataset_label.shape:
            raise ValueError("Input and label shapes must match")

    print(f"Number of samples: {D}")
    inf_time_list = []
    predictions = np.empty(input_shape, dtype=np.float32)

    # === Warming phase
    for _ in range(warming_samples):
        if is_ce_vit:
            for hi, di, xi in zip(h_inputs, d_inputs, dataset_input):
                np.copyto(hi, xi[1])  # sample 1
            _ = do_inference(context, *h_inputs, *d_inputs, h_output, d_output, stream)
        else:
            np.copyto(h_input, dataset_input[1])
            _ = do_inference(context, h_input, d_input, h_output, d_output, stream)

    # === Inference loop
    for i in range(D):
        if i % print_interval == 0 and print_processing_sample:
            print(f"Processing sample {i}")

        if is_ce_vit:
            for hi, di, xi in zip(h_inputs, d_inputs, dataset_input):
                np.copyto(hi, xi[i])
            start = time.time()
            output = do_inference(context, *h_inputs, *d_inputs, h_output, d_output, stream)
            end = time.time()
        else:
            np.copyto(h_input, dataset_input[i])
            start = time.time()
            output = do_inference(context, h_input, d_input, h_output, d_output, stream)
            end = time.time()

        # Record inference time (skip warmup)
        if i > warming_samples:
            inf_time_list.append(end - start)

        # Store output
        output_reshaped = output.reshape(
            (input_shape[1], input_shape[2], input_shape[3]) if CNN_2D
            else (input_shape[1], input_shape[2])
        )
        predictions[i] = output_reshaped

        if break_sample > 0 and i == break_sample:
            break

    # === Compute final stats
    #opt_mse = mean_squared_error(predictions.reshape(D, -1), dataset_label.reshape(D, -1))

    opt_mse = compute_mse_nmse_nmse_db(predictions.reshape(D, -1), dataset_label.reshape(D, -1))[2]

    return (
        opt_mse,
        np.sum(inf_time_list),
        np.mean(inf_time_list),
        np.std(inf_time_list),
        np.min(inf_time_list),
        np.max(inf_time_list),
        inf_time_list
    )


def compute_mse_nmse_nmse_db(predictions: np.ndarray, labels: np.ndarray):
    """
    Computes the Normalized Mean Squared Error (NMSE) and its value in decibels (dB)
    over the full dataset.

    Args:
        predictions (np.ndarray): Predicted values of shape (D, ...) or (N,).
        labels (np.ndarray): Ground truth values of same shape as predictions.

    Returns:
        mse (float): Mean Squared Error.
        nmse (float): Normalized Mean Squared Error.
        nmse_db (float): NMSE in decibel scale.
    """
    pred_flat = predictions.reshape(-1)
    label_flat = labels.reshape(-1)

    mse = mean_squared_error(label_flat, pred_flat)
    signal_power = np.mean(label_flat ** 2)
    nmse = mse / signal_power
    nmse_db = 10 * np.log10(nmse)

    return mse, nmse, nmse_db
    
def run_and_analyze_opt(num_inferences, *args, **kwargs):
    # Lists to store function outputs
    mse_list = []
    inf_time_total_list = []
    inf_time_mean_list = []
    inf_time_std_list = []
    inf_time_min_list = []
    inf_time_max_list = []
    inf_time_outputs_list = []

    # Unpack required inputs
    dataset_input = args[0]
    dataset_label = args[1]
    remaining_args = args[2:]  # h_input(s), d_input(s)

    # Use tqdm for progress tracking
    for _ in tqdm(range(num_inferences), desc="Running inference"):
        opt_mse, opt_inf_time_total, opt_inf_time_mean, opt_inf_time_std, opt_inf_time_min, opt_inf_time_max, opt_inf_list = mean_square_error_and_inf_time(
            dataset_input,
            dataset_label,
            *remaining_args,
            h_output=kwargs['h_output'],
            d_output=kwargs['d_output'],
            stream=kwargs['stream'],
            context=kwargs['context'],
            CNN_2D=kwargs.get('CNN_2D', False),
            print_interval=kwargs.get('print_interval', 50),
            warming_samples=kwargs.get('warming_samples', 50),
            print_processing_sample=kwargs.get('print_processing_sample', False)
        )

        mse_list.append(opt_mse)
        inf_time_total_list.append(opt_inf_time_total)
        inf_time_mean_list.append(opt_inf_time_mean)
        inf_time_std_list.append(opt_inf_time_std)
        inf_time_min_list.append(opt_inf_time_min)
        inf_time_max_list.append(opt_inf_time_max)
        inf_time_outputs_list.append(opt_inf_list)

    def get_stats(lst):
        return {
            'mean': np.mean(lst),
            'max': np.max(lst),
            'min': np.min(lst),
            'std': np.std(lst)
        }

    return {
        'mse_stats': get_stats(mse_list),
        'inf_time_total_stats': get_stats(inf_time_total_list),
        'inf_time_mean_stats': get_stats(inf_time_mean_list),
        'inf_time_std_stats': get_stats(inf_time_std_list),
        'inf_time_min_stats': get_stats(inf_time_min_list),
        'inf_time_max_stats': get_stats(inf_time_max_list),
        'inf_time_outputs_list': inf_time_outputs_list
    }


def optimize_and_get_inference_time(non_opt_model, model_name, test_data, test_label, num_inferences=5, opset=13, CNN_2D=False):
    """
    Convert a TensorFlow model to an optimized TensorRT model and measure inference time.
    Handles both single-input models and multi-input models like Ce-ViT.
    """

    # === Step 1: Convert Model to ONNX ===
    onnx_model_path = f"{model_name}.onnx"
    is_ce_vit = isinstance(test_data, list) and len(test_data) == 4

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found. Converting {model_name} to ONNX...")
        if is_ce_vit:
            spec = (
                tf.TensorSpec((None, 612, 14, 2), tf.float32, name="H_int"),
                tf.TensorSpec((None, 1), tf.float32, name="SNR"),
                tf.TensorSpec((None, 1), tf.float32, name="Doppler"),
                tf.TensorSpec((None, 1), tf.float32, name="Delay"),
            )
        else:
            spec = (tf.TensorSpec((None, *non_opt_model.input.shape[1:]), tf.float32, name="input"),)

        _ = tf2onnx.convert.from_keras(non_opt_model, input_signature=spec, opset=opset, output_path=onnx_model_path)
    else:
        print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")

    # === Step 2: Build TensorRT Engine ===
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model_file:
        model_data = model_file.read()

    if not parser.parse(model_data):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX model parsing failed!")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()

    # Define input shapes
    if is_ce_vit:
        profile.set_shape("H_int", (1, 612, 14, 2), (1, 612, 14, 2), (1, 612, 14, 2))
        profile.set_shape("SNR", (1, 1), (1, 1), (1, 1))
        profile.set_shape("Doppler", (1, 1), (1, 1), (1, 1))
        profile.set_shape("Delay", (1, 1), (1, 1), (1, 1))
    else:
        profile.set_shape("input", (1, 612, 14, 2), (1, 612, 14, 2), (1, 612, 14, 2))

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    # === Step 3: Allocate Memory ===
    h_inputs, d_inputs, h_output, d_output, bindings = allocate_trt_io_memory(engine, test_data)
    stream = cuda.Stream()

    # === Step 4: Prepare Input Format ===
    if is_ce_vit:
        inputs = [arr.astype(np.float32) for arr in test_data]
    else:
        inputs = test_data.astype(np.float32)
    
    # === Step 5: Run Inference and Analyze ===
    if isinstance(h_inputs, list) and len(h_inputs) == 4:
        result = run_and_analyze_opt(
            num_inferences,
            inputs,
            test_label,
            *h_inputs,
            *d_inputs,
            h_output=h_output,
            d_output=d_output,
            stream=stream,
            context=context,
            CNN_2D=CNN_2D,
            print_processing_sample=False
        )
    else:
        result = run_and_analyze_opt(
            num_inferences,
            inputs,
            test_label,
            h_inputs[0],  # pass as ndarray
            d_inputs[0],  # pass as ndarray
            h_output=h_output,
            d_output=d_output,
            stream=stream,
            context=context,
            CNN_2D=CNN_2D,
            print_processing_sample=False
        )

    # === Step 6: Return Timing and MSE Stats ===
    avg_inf_time_ms = np.mean(result['inf_time_outputs_list'])
    min_inf_time_ms = np.min(result['inf_time_outputs_list'])
    max_inf_time_ms = np.max(result['inf_time_outputs_list'])
    std_inf_time_ms = np.std(result['inf_time_outputs_list'])
    inf_time_list_ms = np.array(result['inf_time_outputs_list'])
    opt_mse = result['mse_stats']['mean']


    return [opt_mse, avg_inf_time_ms, min_inf_time_ms, max_inf_time_ms, std_inf_time_ms, inf_time_list_ms]

def allocate_trt_io_memory(engine, test_data):
    """
    Allocates host and device memory for TensorRT I/O bindings, supporting both single and multi-input models.
    
    Parameters:
    - engine: TensorRT engine
    - test_data: Either a single NumPy array (standard models) or list of NumPy arrays (e.g., Ce-ViT)

    Returns:
    - h_inputs: list of host input arrays
    - d_inputs: list of device input arrays
    - h_output: host output array
    - d_output: device output buffer
    - bindings: list of pointers for TensorRT bindings
    """

    h_inputs = []
    d_inputs = []
    bindings = []

    is_ce_vit = isinstance(test_data, list)

    num_inputs = 4 if is_ce_vit else 1

    for i in range(num_inputs):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))

        if is_ce_vit:
            shape = tuple(test_data[i].shape[1:])
        else:
            shape = tuple(test_data.shape[1:])

        full_shape = (1,) + shape
        h_input = cuda.pagelocked_empty(full_shape, dtype=dtype)
        d_input = cuda.mem_alloc(h_input.nbytes)

        h_inputs.append(h_input)
        d_inputs.append(d_input)
        bindings.append(int(d_input))

    # Output buffer
    output_index = engine.num_bindings - 1
    output_name = engine.get_binding_name(output_index)
    output_shape = tuple(engine.get_binding_shape(output_index))
    output_dtype = trt.nptype(engine.get_binding_dtype(output_index))
    h_output = cuda.pagelocked_empty((int(np.prod(output_shape)),), dtype=output_dtype)
    d_output = cuda.mem_alloc(h_output.nbytes)
    bindings.append(int(d_output))

    return h_inputs, d_inputs, h_output, d_output, bindings

def get_inference_optimized_models(models_dict, test_data, test_data_HELENA, test_data_CE_ViT, test_label, num_inferences=5, CNN_2D=False):
    """
    Optimizes multiple models and measures inference time for each.

    Parameters:
    - models_dict: Dictionary where keys are model names (str) and values are non-optimized TensorFlow models.
    - test_data: Input test data (NumPy array) for inference timing.
    - num_samples: Number of samples to run inference on (default: 100).

    Returns:
    - results_dict: Dictionary where keys are model names and values are arrays containing:
        [avg_inf_time_ms, min_inf_time_ms, max_inf_time_ms, std_inf_time_ms, inf_time_list]
    """

    results_dict = {}

    for model_name, model in models_dict.items():
        print(f"Optimizing and measuring inference time for: {model_name}...")

        # Run optimization and measure inference time
        if model_name in ['HELENA', 'LSiDNN 48', 'LSiDNN 1024', 'HELENA MHSA']:
            result = optimize_and_get_inference_time(model, model_name, test_data_HELENA, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)
        elif model_name == "CE-ViT":
            result = optimize_and_get_inference_time(model, model_name, [test_data] + test_data_CE_ViT, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)
        else:
            result = optimize_and_get_inference_time(model, model_name, test_data, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)

        # Store result in dictionary
        results_dict[model_name] = result


    return results_dict

def do_inference(context, *args):
    """
    Handles both single-input and 4-input (CE-ViT) inference.
    Arguments after context should be:
    - For single-input models: h_input, d_input, h_output, d_output, stream
    - For Ce-ViT models: h1, h2, h3, h4, d1, d2, d3, d4, h_output, d_output, stream
    """

    # For Ce-ViT: 4 h_inputs + 4 d_inputs + h_output + d_output + stream = 11 args
    if len(args) == 11:
        h_inputs = args[0:4]
        d_inputs = args[4:8]
        h_output = args[8]
        d_output = args[9]
        stream = args[10]
        bindings = []

        for h_input, d_input in zip(h_inputs, d_inputs):
            cuda.memcpy_htod_async(d_input, h_input, stream)
            bindings.append(int(d_input))

        bindings.append(int(d_output))

        engine = context.engine
        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        return h_output

    # For single-input models
    elif len(args) == 5:
        h_input, d_input, h_output, d_output, stream = args
        bindings = [int(d_input), int(d_output)]

        engine = context.engine
        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        return h_output

    else:
        raise ValueError(f"Unexpected number of arguments for do_inference: {len(args)}")

def compute_flops_mse_inference(
    models_dict, results, test_data, test_data_HELENA, test_data_CE_ViT, test_label, num_inferences=5, CNN_2D=False, json_filename="flops_mse_inf_data.json"
):
    """
    Computes FLOPS, MSE, and inference time for models, saves data to JSON, and returns the results dictionary.
    """
    optimized_inference_results = get_inference_optimized_models(models_dict, test_data, test_data_HELENA, test_data_CE_ViT, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)
    model_names = list(models_dict.keys())
    flops_dict = {}

    random_index = np.random.randint(0, test_data.shape[0])
    
    for model_name, model in models_dict.items():
        if model_name == "CE-ViT":
            test_input =[
                np.expand_dims(test_data[random_index], axis=0),         # (1, 612, 14, 2)
                np.expand_dims(test_data_CE_ViT[0][random_index], axis=0),        # (1, 1)
                np.expand_dims(test_data_CE_ViT[1][random_index], axis=0),    # (1, 1)
                np.expand_dims(test_data_CE_ViT[2][random_index], axis=0)       # (1, 1)
            ]
            
            # Predict
            model(test_input)
        else:
            test_input = np.expand_dims(test_data[random_index], axis=0)
            model(test_input)
        
        flops = get_flops(model)
        flops_dict[model_name] = flops
        print(f'Total FLOPS for {model_name}: {flops}')

    flops_values = [flops_dict[model] / 10.0**9 for model in model_names]
    mse_values = [results["general_results"][model]['mse'] for model in model_names]
    mse_values_opt = [optimized_inference_results[model][0] for model in model_names]
    inf_time_values = [optimized_inference_results[model][1] for model in model_names]

    data_dict = {
        "models": model_names,
        "flops": flops_values,
        "mse": mse_values,
        "optimized_mse": mse_values_opt,
        "inference_time": inf_time_values
    }
    
    with open(json_filename, "w") as json_file:
        json.dump(data_dict, json_file, indent=4)
    
    return data_dict

def plot_flops_mse_inference(
    data_dict, filename="flops_mse_inf_plot.png",
    plot_flops=True, plot_mse=True, plot_mse_opt=True, plot_inf_time=True,
    title_font_size=18, tick_font_size=14, legend_font_size=14, label_font_size=16, round_val=3):
    """
    Plots FLOPS, MSE, optimized MSE, and inference time using precomputed data with optional selection.
    Ensures no empty spaces when some bars are deactivated. Keeps legend and value labels inside the figure.
    """
    model_names = data_dict["models"]
    flops_values = data_dict["flops"]
    mse_values = data_dict["mse"]
    mse_values_opt = data_dict["optimized_mse"]
    inf_time_values = data_dict["inference_time"]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    spacing_factor_bar_group = 2  # Increase to make more space
    index = np.arange(len(model_names)) * spacing_factor_bar_group
    #index = np.arange(len(model_names))

    # Determine active metrics and their positions
    active_metrics = []
    if plot_mse:
        active_metrics.append(("MSE Original Model", mse_values, ax2, 'orange', '//'))
    if plot_mse_opt:
        active_metrics.append(("MSE Optimized Model", mse_values_opt, ax2, 'red', '\\'))
    if plot_flops:
        active_metrics.append(("FLOPS", flops_values, ax1, 'yellow', 'xx'))
    if plot_inf_time:
        active_metrics.append(("Inference Time", inf_time_values, ax2, 'blue', '..'))

    num_active = len(active_metrics)
    if num_active == 0:
        print("No plots enabled. Enable at least one metric.")
        return

    # Adjust bar width to create space between bars
    total_bar_width = 0.8  # Reduce total width to leave space
    bar_width = min(total_bar_width / num_active, 0.25)  # Ensure bars fit within category

    # Adjust bar positions dynamically with spacing
    spacing_factor = 2  # Increase spacing between bars
    offsets = np.linspace(-bar_width * (num_active - 1) * spacing_factor / 2,
                          bar_width * (num_active - 1) * spacing_factor / 2,
                          num_active)

    bars = []
    for (label, values, ax, color, hatch), offset in zip(active_metrics, offsets):
        bars.append(ax.bar(index + offset, values, bar_width, label=label, color=color, edgecolor='black', hatch=hatch))

    # Ensure value labels stay inside the figure
    max_values = []  # Store max values for scaling y-limits
    for bar_group in bars:
        for bar in bar_group:
            yval = bar.get_height()
            max_values.append(yval)
            ax = bar.axes  # Get the corresponding axis
            max_ylim = ax.get_ylim()[1]  # Get upper limit of the axis

            # Place label inside the bar if it's too close to the top
            label_ypos = yval if yval < 0.95 * max_ylim else 0.95 * max_ylim
            ax.text(bar.get_x() + bar.get_width() / 2, label_ypos, f"{round(yval, round_val)}",
                    ha='center', va='bottom', fontsize=tick_font_size, rotation=70, color='black')

    # **Increase y-axis limits to avoid crowding at the top**
    if max_values:
        max_bar_height = max(max_values)
        ax1.set_ylim(0, max_bar_height * 1.2)  # Increase by 20%
        ax2.set_ylim(0, max_bar_height * 1.2)  # Ensure second y-axis is also adjusted

    # Labels and Titles
    ax1.set_xlabel('Channel Estimation Model', fontsize=label_font_size)
    ax1.set_ylabel('FLOPS ($10^9$)', fontsize=label_font_size)
    if plot_inf_time:
        ax2.set_ylabel('MSE ($10^{-3}$) & Inference Time (ms)', fontsize=label_font_size)
    else:
        ax2.set_ylabel('MSE ($10^{-3}$)', fontsize=label_font_size)


    # Adjust x-ticks to align with bars
    ax1.set_xticks(index)
    ax1.set_xticklabels(model_names, fontsize=tick_font_size, rotation=45, ha='right')

    # Add legend inside the figure (top-right, adjusted to stay inside the frame)
    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.8), fontsize=legend_font_size, frameon=True)

    # Adjust top margin to prevent labels from being cut off
    plt.subplots_adjust(top=0.8)  # Reduces top margin to allow space

    # Save and show the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_flops_nmse_inference(
    data_dict, filename="flops_nmse_inf_plot.png",
    plot_flops=True, plot_nmse=True, plot_nmse_opt=True, plot_inf_time=True,
    title_font_size=18, tick_font_size=14, legend_font_size=14, label_font_size=16, round_val=2):
    """
    Plots FLOPS, NMSE (in dB), optimized NMSE, and inference time using precomputed data with optional selection.
    Ensures no empty spaces when some bars are deactivated. Keeps legend and value labels inside the figure.
    """

    model_names = data_dict["models"]
    flops_values = data_dict["flops"]
    nmse_values = data_dict["mse"]  # in dB (e.g., -17.4)
    nmse_values_opt = data_dict["optimized_mse"]
    inf_time_values = data_dict["inference_time"]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    spacing_factor_bar_group = 2
    index = np.arange(len(model_names)) * spacing_factor_bar_group

    active_metrics = []
    if plot_nmse:
        active_metrics.append(("−NMSE Original", nmse_values, ax2, 'orange', '//'))
    if plot_nmse_opt:
        active_metrics.append(("−NMSE Optimized", nmse_values_opt, ax2, 'red', '\\'))
    if plot_flops:
        active_metrics.append(("FLOPS", flops_values, ax1, 'yellow', 'xx'))
    if plot_inf_time:
        active_metrics.append(("Inference Time", inf_time_values, ax2, 'blue', '..'))

    num_active = len(active_metrics)
    if num_active == 0:
        print("No plots enabled. Enable at least one metric.")
        return

    total_bar_width = 0.8
    bar_width = min(total_bar_width / num_active, 0.25)
    spacing_factor = 2
    offsets = np.linspace(-bar_width * (num_active - 1) * spacing_factor / 2,
                          bar_width * (num_active - 1) * spacing_factor / 2,
                          num_active)

    bars = []
    for (label, values, ax, color, hatch), offset in zip(active_metrics, offsets):
        bars.append(ax.bar(index + offset, values, bar_width, label=label, color=color, edgecolor='black', hatch=hatch))

    max_values = []
    for bar_group in bars:
        for bar in bar_group:
            yval = bar.get_height()
            max_values.append(abs(yval))
            ax = bar.axes
            max_ylim = ax.get_ylim()[1]
            label_ypos = yval if yval < 0.95 * max_ylim else 0.95 * max_ylim
            ax.text(bar.get_x() + bar.get_width() / 2, label_ypos, f"{round(yval, round_val)}",
                    ha='center', va='bottom', fontsize=tick_font_size, rotation=70, color='black')

    if max_values:
        max_bar_height = max(max_values)
        ax1.set_ylim(0, max_bar_height * 1.2)
        ax2.set_ylim(-max_bar_height * 1.2, 0)  # For dB scale (e.g., from -30 to 0)

    ax1.set_xlabel('Channel Estimation Model', fontsize=label_font_size)
    ax1.set_ylabel('FLOPS ($10^9$)', fontsize=label_font_size)
    if plot_inf_time:
        ax2.set_ylabel('−NMSE (dB) & Inference Time (ms)', fontsize=label_font_size)
    else:
        ax2.set_ylabel('−NMSE (dB)', fontsize=label_font_size)

    ax1.set_xticks(index)
    ax1.set_xticklabels(model_names, fontsize=tick_font_size, rotation=45, ha='right')

    fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.8), fontsize=legend_font_size, frameon=True)
    plt.subplots_adjust(top=0.8)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_radar_nmse_flops_inftime(
    data_dict, filename="radar_nmse_flops_inf_plot.png",
    title_font_size=18, label_font_size=14, legend_font_size=10
):
    """
    Plots a radar chart comparing models across normalized −NMSE (dB), FLOPS, and Inference Time.
    Lower values are better; normalization accounts for this.
    """
    models = data_dict["models"]
    nmse = np.array(data_dict["mse"])           # Assumed in dB (more negative = better)
    flops = np.array(data_dict["flops"])         # Typically in GFLOPS
    inf_time = np.array(data_dict["inference_time"])  # Typically in ms

    def normalize(values, reverse=False):
        if reverse:
            values = -values
        return (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)

    # Normalize each metric: lower is better, so NMSE reversed
    nmse_norm = normalize(nmse, reverse=True)
    flops_norm = normalize(flops)
    inf_time_norm = normalize(inf_time)

    # Radar setup
    labels = ["−NMSE (dB)", "FLOPS", "Inference Time"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Plot each model
    for i, model in enumerate(models):
        values = [nmse_norm[i], flops_norm[i], inf_time_norm[i]]
        values += values[:1]
        ax.plot(angles, values, label=model, linewidth=2)
        ax.fill(angles, values, alpha=0.08)

    # Axis styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=label_font_size)
    ax.set_yticks([])
    ax.set_title("Model Comparison (Normalized Metrics)", fontsize=title_font_size, pad=20)

    # Legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=legend_font_size)

    # Save and show
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_bubble_tradeoff(data_dict, filename="bubble_tradeoff_plot.png", dpi=300):
    """
    Bubble chart: −NMSE vs. Inference Time (ms), bubble size = FLOPs.
    Ensures all bubbles fit and labels are placed above each bubble.
    """
    model_names = data_dict["models"]
    flops = np.array(data_dict["flops"])
    nmse = np.array(data_dict["optimized_mse"])
    inference_time = np.array(data_dict["inference_time"]) * 1000  # convert s → ms

    # Normalize FLOPs to scale bubbles, add minimum size to ensure visibility
    flops_scaled = flops / flops.max() * 10000
    flops_scaled = np.clip(flops_scaled, 50, None)  # Ensure a minimum size


    # Dynamic adjustment for margin based on max bubble size
    margin_x = inference_time.max() * 0.15
    #margin_y = abs(nmse.max() - nmse.min()) * 0.15

    # Create plot
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        inference_time,
        -nmse,
        s=flops_scaled,
        alpha=0.6,
        c=flops_scaled,
        cmap='viridis',
        edgecolors='black',
        linewidth=0.5
    )

    # Add model name above or below each bubble with custom offset control
    for i, name in enumerate(model_names):
        radius_pts = np.sqrt(flops_scaled[i])
        
        # Default offset factor (scales with bubble radius)
        offset_factor = 0.5
    
        # Reduce offset for specific models
        if name in ["ChannelNet", "EDSR"]:
            offset_factor = 0.4  # bring label closer
    
        offset = (radius_pts / 72) * offset_factor  # points → data coords (approx)
    
        if name == "ChannelNet":
            va = 'top'
            y_pos = -nmse[i] - offset  # BELOW
        else:
            va = 'bottom'
            y_pos = -nmse[i] + offset  # ABOVE
    
        plt.text(
            inference_time[i],
            y_pos,
            name,
            fontsize=11,
            ha='center',
            va=va
    )

    plt.xlabel("Inference Time (ms)", fontsize=16)
    plt.ylabel("−NMSE (dB)", fontsize=16)
    #plt.title("Model Trade-off: Accuracy vs. Inference Time vs. Complexity", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Color bar for FLOPs
    cbar = plt.colorbar(scatter)
    cbar.set_label('Model Complexity (10^6 FLOPs)', fontsize=16)

    # Adjust axis limits to ensure all bubbles + labels fit
    #plt.xlim(0, inference_time.max() + margin_x)
    # Ensure x-axis goes to at least 0.5 ms and label is shown
    x_max = max(0.5, inference_time.max() * 1.1)
    plt.xlim(0, x_max)

    #plt.ylim(-nmse.min() - margin_y, -nmse.max() + margin_y)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.show()


def compute_mse_nmse_nmse_db(true, pred):
    mse = np.mean((true - pred) ** 2)
    power = np.mean(true ** 2)
    nmse = mse / power
    nmse_db = 10 * np.log10(nmse)
    return mse, nmse, nmse_db
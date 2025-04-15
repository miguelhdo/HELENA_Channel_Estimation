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


def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as f:
        # List all groups
        keys = list(f.keys())

        # Access the specific dataset containing object references
        obj_ref_dataset = f[keys[1]]

        # Variables to store the loaded data
        trainData = None
        trainLabels = None
        trainPractical = None
        trainLinearInterpolation = None
        otherLabels = None

        # Iterate through each object reference in the dataset
        for i, obj_ref in enumerate(obj_ref_dataset):
            # Dereference the object reference to access the actual object
            obj = f[obj_ref[0]]

            if i == 0:
                trainData = np.array(obj, dtype=obj.dtype)
                trainData = np.transpose(trainData, (0, 3, 2, 1))
            elif i == 1:
                trainLabels = np.array(obj, dtype=obj.dtype)
                trainLabels = np.transpose(trainLabels, (0, 3, 2, 1))
            elif i == 2:
                trainPractical = np.array(obj, dtype=obj.dtype)
                trainPractical = np.transpose(trainPractical, (0, 3, 2, 1))
            elif i == 3:
                trainLinearInterpolation = np.array(obj, dtype=obj.dtype)
                trainLinearInterpolation = np.transpose(trainLinearInterpolation, (0, 3, 2, 1))
            else:
                otherLabels = np.array(obj, dtype=obj.dtype)
                # profileIdx, SNRdB, delaySpread, dopplerShift

    return trainData, trainLabels, trainPractical, trainLinearInterpolation, otherLabels


def split_dataset(trainData, trainLabels, trainPractical, trainLinearInterpolation, otherLabels, val_test_size=0.3, random_state=42):
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
    otherLabels_train = otherLabels[train_indices]

    trainData_validation = trainData[validation_indices]
    trainLabels_validation = trainLabels[validation_indices]
    trainPractical_validation = trainPractical[validation_indices]
    trainLinearInterpolation_validation = trainLinearInterpolation[validation_indices]
    otherLabels_validation = otherLabels[validation_indices]

    trainData_test = trainData[test_indices]
    trainLabels_test = trainLabels[test_indices]
    trainPractical_test = trainPractical[test_indices]
    trainLinearInterpolation_test = trainLinearInterpolation[test_indices]
    otherLabels_test = otherLabels[test_indices]
    print("Training, Validation, and Test split done!")

    return (trainData_train, trainLabels_train, trainPractical_train, trainLinearInterpolation_train, otherLabels_train,
            trainData_validation, trainLabels_validation, trainPractical_validation, trainLinearInterpolation_validation, otherLabels_validation,
            trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, otherLabels_test)


def plot_mse_vs_snr(results, otherLabels_test, trainLabels_test, filename="MSE_vs_SNR.png",
                    title_font_size=18, tick_font_size=14, label_font_size=16, legend_font_size=14):
    # Unique SNR values
    unique_snr_values, unique_snr_counts = np.unique(otherLabels_test[:, 1], return_counts=True)

    print(unique_snr_values)

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
        indices = otherLabels_test[:, 1] == snr
        first_dimension = idx

        # Store results for SNR, practical, and linear interpolation
        mse_results["SNR"].append(snr)

        # Dynamically calculate MSE for each model in results
        for model_name in model_names:
            mse_model = mean_squared_error(
                trainLabels_test[indices].reshape(first_dimension, -1),
                results["general_results"][model_name]["predictions"][indices].reshape(first_dimension, -1)
            )
            mse_results[f"MSE_{model_name}"].append(mse_model)

    # Plotting the main figure
    plt.figure(figsize=(10, 6))

    # Dynamically plot all models from the results
    markers = ['o', '>', 'v', 'p', 'd', 's', 'x', '+', '<', '*']  # Different markers for dynamic models
    colors = ['red', 'purple', 'orange', 'brown', 'cyan', 'blue', 'green', 'fuchsia', 'black', 'greenyellow', 'crimson', 'chocolate']  # Different colors for dynamic models
    linestyles = [':', '-', '--', '-.', '-', '--', '-.']  # Different line styles for dynamic models

    for idx, model_name in enumerate(model_names):
        plt.plot(mse_results["SNR"], mse_results[f"MSE_{model_name}"],
                 label=f"MSE {model_name}", marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)], linestyle=linestyles[idx % len(linestyles)])

    # Set labels and title with configurable font sizes
    plt.xlabel("SNR (dB)", fontsize=label_font_size)
    plt.ylabel("MSE (log10)", fontsize=label_font_size)
    plt.yscale('log')
    plt.legend(fontsize=legend_font_size)
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
    ax_inset.set_ylim(0.0023, 0.004)  # Adjusted y-limits
    ax_inset.set_yscale('log')
    ax_inset.grid(True)

    # Remove axis labels in the inset
    ax_inset.tick_params(axis='y', which='both', labelleft=False)

    # Set tick parameters for inset tick labels
    ax_inset.tick_params(axis='both', which='major', labelsize=tick_font_size)

    # Save the figure as a PNG image
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')

    plt.show()



# Function to calculate FLOPS
def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(tf.TensorSpec([1] + model.inputs[0].shape[1:], model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    frozen_func.graph.as_graph_def()

    # Calculate FLOPS
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                          run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops

def plot_flops_and_mse(models_dict, dummy_input, results, filename="flops_mse_plot.png", title_font_size=18, tick_font_size=14, legend_font_size=14, label_font_size=16):
    flops_dict = {}

    # Loop through each model in models_dict and calculate FLOPs
    for model_name, model in models_dict.items():
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


import matplotlib.pyplot as plt
import numpy as np

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

def single_sample_experiment(random_index, trainData_test, trainLabels_test, trainPractical_test, trainLinearInterpolation_test, loaded_models, results, verbose = 0):
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
        predicted_sample = model.predict(np.expand_dims(input_sample, axis=0))[0]

        # Store the prediction
        results["single_sample_experiment"][random_index]["predictions"][model_name] = predicted_sample

        # Compute and store the magnitude
        results["single_sample_experiment"][random_index]["magnitudes"][model_name] = np.linalg.norm(predicted_sample, axis=-1)

        # Compute and store the MSE between the label and the predicted sample
        mse = np.mean((label_sample - predicted_sample) ** 2)
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
    results["single_sample_experiment"][random_index]["mse"]["LS"] = np.mean((label_sample - practical_sample) ** 2)
    results["single_sample_experiment"][random_index]["mse"]["LI"] = np.mean((label_sample - linearInterpol_sample) ** 2)

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

def mean_square_error_and_inf_time(dataset_input, dataset_label, h_input, d_input, h_output, d_output, stream, context, CNN_2D=False, break_sample=0, print_interval=50, warming_samples=50, print_processing_sample=False):
    """
    Calculate the mean square error and inference time statistics for a dataset with shape [D, 612, 14, 2].

    Parameters:
    dataset_input (numpy array): The input values with shape [D, 612, 14, 2].
    dataset_label (numpy array): The true values with shape [D, 612, 14, 2].
    break_sample (int): Number of samples after which to break the loop (0 means use all samples).
    print_interval (int): Print progress every print_interval samples.
    warming_samples (int): Number of warming samples to run before measuring time.

    Returns:
    tuple: (mse_mean, mse_std, inf_time_mean, inf_time_std)
           Mean and standard deviation of the mean squared error and inference time.
    """
    if dataset_input.shape != dataset_label.shape:
        raise ValueError("The shapes of the input and label datasets must be the same.")

    D = dataset_input.shape[0]
    print(f"Number of samples: {D}")
    inf_time_list = []
    predictions = np.empty(dataset_input.shape)

    # Warming phase to stabilize GPU performance
    for i in range(warming_samples):
        np.copyto(h_input, dataset_input[1])
        _ = do_inference(context, h_input, d_input, h_output, d_output, stream)

    # Inference and error calculation
    for i in range(D):
        if i == 1:
            print("Warming up")
        if i % print_interval == 0 and print_processing_sample:
            print(f"Processing sample {i}")

        start_time = time.time()

        # Copy input data and run inference
        np.copyto(h_input, dataset_input[i])

        if i == 100:
            print("Inference time")
        if i > 100:
            start_time = time.time()
            output = do_inference(context, h_input, d_input, h_output, d_output, stream)
            end_time = time.time()
            inference_time = end_time - start_time
            inf_time_list.append(inference_time)
        else:
            output = do_inference(context, h_input, d_input, h_output, d_output, stream)

        # Reshape and store the output
        if CNN_2D:
            dataset_output = output.reshape((dataset_input.shape[1],dataset_input.shape[2], dataset_input.shape[3]))
        else:
            dataset_output = output.reshape(dataset_input.shape[1], dataset_input.shape[2])

        predictions[i] = dataset_output

        if i == break_sample and break_sample > 0:
            break

    opt_mse = mean_squared_error(predictions.reshape(dataset_input.shape[0],-1), dataset_label.reshape(dataset_input.shape[0],-1))

    # Calculate mean and standard deviation for MSE and inference time
    opt_inf_time_total = np.sum(inf_time_list)
    opt_inf_time_mean = np.mean(inf_time_list)
    opt_inf_time_std = np.std(inf_time_list)
    opt_inf_time_max = np.max(inf_time_list)
    opt_inf_time_min = np.min(inf_time_list)

    return opt_mse, opt_inf_time_total, opt_inf_time_mean, opt_inf_time_std, opt_inf_time_min, opt_inf_time_max, inf_time_list

def run_and_analyze_opt(num_inferences, *args, **kwargs):
    # Lists to store function outputs
    mse_list = []
    inf_time_total_list = []
    inf_time_mean_list = []
    inf_time_std_list = []
    inf_time_min_list = []
    inf_time_max_list = []
    inf_time_outputs_list = []

    # Use tqdm for progress tracking
    for _ in tqdm(range(num_inferences), desc="Running inference"):
        # Run the function and collect outputs
        opt_mse, opt_inf_time_total, opt_inf_time_mean, opt_inf_time_std, opt_inf_time_min, opt_inf_time_max, opt_inf_list = mean_square_error_and_inf_time(
            *args, **kwargs
        )

        # Append each output to the respective list
        mse_list.append(opt_mse)
        inf_time_total_list.append(opt_inf_time_total)
        inf_time_mean_list.append(opt_inf_time_mean)
        inf_time_std_list.append(opt_inf_time_std)
        inf_time_min_list.append(opt_inf_time_min)
        inf_time_max_list.append(opt_inf_time_max)
        inf_time_outputs_list.append(opt_inf_list)

    # Function to compute stats
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

    Parameters:
    - non_opt_model: TensorFlow/Keras model (not optimized)
    - test_data: Input test data (NumPy array)
    - num_inferences: Number of times to run inference (default: 100)
    - opset: ONNX opset version (default: 13)

    Returns:
    - avg_inf_time_ms: Average inference time in milliseconds
    - min_inf_time_ms: Minimum inference time in milliseconds
    - max_inf_time_ms: Maximum inference time in milliseconds
    - std_inf_time_ms: Standard deviation of inference time in milliseconds
    - inf_time_list: List of individual inference times in milliseconds
    """

    # Step 1: Convert Model to ONNX
    #onnx_model_path = f'{model_name}.onnx'
    #spec = (tf.TensorSpec((None, *non_opt_model.input.shape[1:]), tf.float32, name="input"),)
    #_, _ = tf2onnx.convert.from_keras(non_opt_model, input_signature=spec, opset=opset, output_path=onnx_model_path)

    # Ensure ONNX model does not contain INT64 weights (since TensorRT does not support it)
    #convert_int64_to_int32(onnx_model_path, onnx_model_path)  # Ensure this function is properly defined elsewhere

    # Step 2: Load ONNX Model into TensorRT
    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    #builder = trt.Builder(TRT_LOGGER)
    #network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    #parser = trt.OnnxParser(network, TRT_LOGGER)

    #with open(onnx_model_path, 'rb') as model_file:
    #    model_data = model_file.read()


    # Step 1: Convert Model to ONNX (only if it does not already exist)
    onnx_model_path = f'{model_name}.onnx'

    if not os.path.exists(onnx_model_path):
        print(f"ONNX model not found. Converting {model_name} to ONNX...")
        spec = (tf.TensorSpec((None, *non_opt_model.input.shape[1:]), tf.float32, name="input"),)
        _, _ = tf2onnx.convert.from_keras(non_opt_model, input_signature=spec, opset=opset, output_path=onnx_model_path)
    else:
        print(f"ONNX model already exists: {onnx_model_path}. Skipping conversion.")

    # Step 2: Load ONNX Model into TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model_file:
        model_data = model_file.read()

    # Ensure ONNX model does not contain INT64 weights (since TensorRT does not support it)
    #convert_int64_to_int32(onnx_model_path, onnx_model_path)  # Ensure this function is properly defined elsewhere

    if not parser.parse(model_data):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("ONNX model parsing failed!")

    # Step 3: Optimize with TensorRT
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # Enable FP16 precision (only if supported by hardware)
    profile = builder.create_optimization_profile()

    # Set the dimensions for the input, change according to your input shape
    min_shape = (1, 612, 14, 2)  # Minimum batch size
    opt_shape = (1, 612, 14, 2)  # Optimum batch size
    max_shape = (1, 612, 14, 2)  # Maximum batch size

    profile.set_shape("input", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    # Allocate host and device memory
    #input_shape = (1, 612, 14, 2)  # Adjust based on your model's input shape
    input_shape = (1, test_data.shape[1], test_data.shape[2], test_data.shape[3])  # Match input shape
    input_size = trt.volume(input_shape) * trt.float32.itemsize

    input_name = engine.get_tensor_name(0)  # assuming first binding is input
    output_name = engine.get_tensor_name(1)  # assuming second binding is output

    output_shape = engine.get_tensor_shape(output_name)
    output_size = trt.volume(output_shape) * trt.float32.itemsize

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    h_input = cuda.pagelocked_empty(input_shape, dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)

    stream = cuda.Stream()

    result = run_and_analyze_opt(num_inferences, test_data, test_label, h_input, d_input, h_output, d_output, stream, context, print_interval=800, CNN_2D=CNN_2D, print_processing_sample=False)

    #opt_mse, opt_inf_time_total, opt_inf_time_mean, opt_inf_time_std, opt_inf_time_min, opt_inf_time_max, inf_time_list= mean_square_error_and_inf_time(test_data, test_label, h_input, d_input, h_output, d_output, stream, context, print_interval=800, CNN_2D=CNN_2D)

    # Compute Statistics
    #avg_inf_time_ms = np.mean(inf_time_list)
    #min_inf_time_ms = np.min(inf_time_list)
    #max_inf_time_ms = np.max(inf_time_list)
    #std_inf_time_ms = np.std(inf_time_list)
    # Compute Statistics
    avg_inf_time_ms = np.mean(np.array(result['inf_time_outputs_list']).flatten())
    min_inf_time_ms = np.min(np.array(result['inf_time_outputs_list']).flatten())
    max_inf_time_ms = np.max(np.array(result['inf_time_outputs_list']).flatten())
    std_inf_time_ms = np.std(np.array(result['inf_time_outputs_list']).flatten())
    inf_time_list_ms = np.array(result['inf_time_outputs_list']).flatten()
    opt_mse = result['mse_stats']['mean']
    return [opt_mse, avg_inf_time_ms, min_inf_time_ms, max_inf_time_ms, std_inf_time_ms, inf_time_list_ms]

def get_inference_optimized_models(models_dict, test_data, test_label, num_inferences=5, CNN_2D=False):
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
        result = optimize_and_get_inference_time(model, model_name, test_data, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)
        #print(result)

        # Store result in dictionary
        results_dict[model_name] = result


    return results_dict

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Setup tensor address
    bindings = bindings=[int(d_input), int(d_output)]

    engine = context.engine
    for i in range(engine.num_io_tensors):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

    # Run inference
    context.execute_async_v3(stream_handle=stream.handle)

    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

def compute_flops_mse_inference(
    models_dict, results, test_input, test_data, test_label, num_inferences=5, CNN_2D=False, json_filename="flops_mse_inf_data.json"
):
    """
    Computes FLOPS, MSE, and inference time for models, saves data to JSON, and returns the results dictionary.
    """
    optimized_inference_results = get_inference_optimized_models(models_dict, test_data, test_label, num_inferences=num_inferences, CNN_2D=CNN_2D)
    model_names = list(models_dict.keys())
    flops_dict = {}

    for model_name, model in models_dict.items():
        model(test_input)
        flops = get_flops(model)
        flops_dict[model_name] = flops
        print(f'Total FLOPS for {model_name}: {flops}')

    flops_values = [flops_dict[model] / 10.0**9 for model in model_names]
    mse_values = [results["general_results"][model]['mse']*1000 for model in model_names]
    mse_values_opt = [optimized_inference_results[model][0]*1000 for model in model_names]
    inf_time_values = [optimized_inference_results[model][1]*1000 for model in model_names]

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

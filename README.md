# HELENA: High-Efficiency Learning-Based Channel Estimation Using Neural Attention

This repository provides the code and scripts necessary to reproduce the experiments and results from the paper: **“HELENA: High-Efficiency Learning-Based Channel Estimation Using Neural Attention.”**

It includes tools for benchmarking the performance of several deep learning models for channel estimation in terms of Normalized Mean Squared Error (NMSE), computational complexity (FLOPS), and inference latency, evaluated using both TensorFlow and TensorRT-optimized models.

---

## 🧪 Quickstart: Running the Evaluation

After cloning the repository, installing the required dependencies, and setting up your environment, run the evaluation script with the following command:

<pre><code>python eval_models.py --input_file models_dataset_input_conf.json --num_inferences 100 --prefix_output_files "V100" --batch_size 1</code></pre>

### 🔧 Script Arguments

- `--input_file`: Path to a JSON file specifying the location of trained Keras model files used for performance comparison.
- `--num_inferences`: Number of inference runs performed with the TensorRT-optimized models (used for accurate latency measurement).
- `--prefix_output_files`: Prefix for output filenames
- `--batch_size`: Batch size used for inference with the original (non-optimized) TensorFlow models.

---

### 📁 Dataset Information

An example dataset containing approximately **5% of the original dataset** (train, validation, and test) is provided for initial code testing and validation purposes.

> **Note:** The full dataset used in the paper will be publicly released in **June 2025**, as the complete test dataset is currently employed in a research assignment with a submission deadline at the end of May 2025.

---

### 📚 Dependencies & Installation

The file `requirements.txt` contains all Python dependencies required to run the experiments and evaluations.

To install the necessary Python libraries, run:

<pre><code>pip install -r requirements.txt</code></pre>

For quick reference, the main libraries used are:

- `tensorflow==2.15.1` (includes `keras==2.15.0`)
- `tensorrt==8.6.1`
- `pycuda==2025.1`
- `tf2onnx==1.16.1`
- `tqdm==4.67.1`

> ⚠️ **Note**: As long as your system already has **CUDA** and **cuDNN** versions compatible with **TensorRT 8.6.1**, installing the above packages should allow the code to run out of the box.
>
### 🚧 Future Updates

Additional scripts with training details for each model will be progressively added to this repository. 




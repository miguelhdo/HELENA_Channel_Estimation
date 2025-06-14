# HELENA: High-Efficiency Learning-Based Channel Estimation Using Neural Attention

This repository provides the code and scripts necessary to reproduce the experiments and results from the paper: **“HELENA: High-Efficiency Learning-Based Channel Estimation Using Neural Attention.”**

It includes tools for benchmarking the performance of several deep learning models for channel estimation in terms of Normalized Mean Squared Error (NMSE), computational complexity (FLOPS), and inference latency, evaluated using both TensorFlow and TensorRT-optimized models.

---

## 🧪 Quickstart: Running the Evaluation

After cloning the repository, installing the required dependencies, and setting up your environment, run the evaluation script with the following command:

<pre><code>python eval_models.py python eval_models.py --input_file models_dataset_input_conf.json --prefix_output_files V100 --num_inferences 100  --batch_size=32</code></pre>

### 🔧 Script Arguments

- `--input_file`: Path to a JSON file specifying the location of trained Keras model files used for performance comparison.
- `--num_inferences`: Number of inference runs performed with the TensorRT-optimized models (used for accurate latency measurement).
- `--prefix_output_files`: Prefix for output filenames
- `--batch_size`: Batch size used for inference with the original (non-optimized) TensorFlow models.

---

### 📁 Dataset Information

The dataset can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15210986.svg)](https://doi.org/10.5281/zenodo.15210986)


The paper describing the dataset can be found here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15665271.svg)](https://doi.org/10.5281/zenodo.15665271)

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

> ⚠️ > **Note**: As long as your system has **CUDA** and **cuDNN** versions compatible with **TensorRT 8.6.1**, installing the required packages should allow the code to run out of the box—**provided that your GPU supports TensorRT acceleration and all model layer types are compatible.**
>
### 🚧 Future Updates

Additional scripts with training details for each model will be progressively added to this repository. 




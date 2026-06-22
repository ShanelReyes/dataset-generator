# Dataset Generation and Preparation Suite for PPDMaaS

This repository contains the official dataset generation and preprocessing pipeline utilized to evaluate the quantum-safe Privacy-Preserving Data Mining as a Service (PPDMaaS) architecture. These tools are designed to facilitate experimental reproducibility, allowing to evaluate cryptographic overhead and platform elasticity under standardized, controlled conditions.

# Dataset Generator for PPDMaaS Scalability 

This repository contains the main scripts used to generate synthetic datasets for performance and scalability evaluations of the post-quantum privacy-preserving data mining as a service (PPDMaaS) platform. 

These tools are provided to ensure full experimental reproducibility and to enable researchers to analyze the scalability behaviors of homomorphic encryption under isolated and controlled data dimensions.

## Repository Structure and Component Descriptions

### 1. `generate_batch_by_size.py`
This script automates the bulk generation of synthetic data using parallel processing.
* **Main function:** Builds datasets for clustering and classification tasks based on a defined size limit in megabytes.
* **Trace Structure:** Automatically generates execution logs associated with conventional and post-quantum algorithms (e.g., `SKMEANS`, `DBSKMEANS`, `SKMEANSPQC`, `SKNNPQC`).
* **Mathematical modeling:** Uses the `make_blobs` and `make_classification` functions from the `scikit-learn` library, applying Gaussian distributions to define the feature matrices ($X$) and label vectors ($y$), which are stored in NumPy binary format (`.npy`).
* **Orchestration:** Uses `ProcessPoolExecutor` for the simultaneous execution of tasks according to the capabilities of the available hardware.

### 2. `datapreparation.py`
This script processes and adapts datasets from the UCI’s actual repository to make them compatible with homomorphic encryption schemes.
* **Main function:** Reads structural configuration from a control file (`datasets_desc3.csv`) to filter and normalize heterogeneous data.
* **Processing routines:** Applies label encoding (`LabelEncoder`), removes text metadata, isolates class variables, and replaces missing or corrupted values with uniform numerical representations.
* **Output format:** Generates headerless CSV files, label vectors, and index counters required by secure computation layers.

### 3. `Generate_datasets.ipynb`
This Jupyter notebook allows for interactive manipulation and pre-validation of data distributions.
* **Profile processing:** Reads a master configuration file (`trace_experiment_schemes.csv`) to synthesize custom structures for each client.
* **Stochastic model:** Implements an exponential distribution to simulate the time intervals between network requests:
  $$f(x) = \lambda e^{-\lambda x}$$
  where $\lambda = 0.5$ represents the average arrival rate used in the service response analysis.
* **In-memory size calculation:** Estimates the physical space occupied by the generated tensors using the standard data type relationship:
  $$\text{Size (Bytes)} = \text{Samples} \times \text{Features} \times \text{Data Type Size (8 Bytes)}$$
* **Visualization:** Uses `matplotlib` and `seaborn` to examine the variable space and the behavior of arrival frequencies using density histograms.

## Environment Configuration

The parameters of the bulk generation script are controlled by environment variables specified in a `.env` file:

```env
BASE_PATH="/rory/sink"        # Path where output files are stored
EXPECTED_SIZE_MB=“2.0”       # Target cumulative size in megabytes
MAX_RECORDS="100"            # Maximum number of records per dataset
MAX_ATTRIBUTES="5"           # Maximum number of attributes per record
CLASSES=“2”                  # Number of classes or centroids (k)
TASK_ID="CLUSTERING"         # Task type: ‘CLUSTERING’ or ‘CLASSIFICATION’
MAX_PROCESS="2"              # Maximum number of parallel processes
```

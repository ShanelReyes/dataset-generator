import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs, make_classification
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from numpy.random import default_rng
import math 
import os
from scipy import stats
from tqdm import tqdm
from dotenv import load_dotenv

ENV_FILE_PATH = os.environ.get("ENV_FILE_PATH",".env")

if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)

BASE_PATH         = os.environ.get("BASE_PATH","/rory/sink")
EXPECTED_SIZE_MB  = float(os.environ.get("EXPECTED_SIZE_MB","2"))
MAX_RECORDS       = int(os.environ.get("MAX_RECORDS","100"))
MAX_ATTRIBUTES    = int(os.environ.get("MAX_ATTRIBUTES","5"))
CLASSES           = int(os.environ.get("CLASSES","2"))
TASK_ID           = os.environ.get("TASK_ID","CLUSTERING")
MAX_PROCESS       = int(os.environ.get("MAX_PROCESS","2"))

DATASET_EXTENSION = os.environ.get("DATASET_EXTENSION","npy")
THRESHOLD         = float(os.environ.get("THRESHOLD","0.004"))
SENS              = float(os.environ.get("SENS","0.000001"))
MAX_ITERATIONS    = int(os.environ.get("MAX_ITERATIONS","3"))
# NUM_CHUNKS        = int(os.environ.get("NUM_CHUNKS","2"))
STD_RECORDS      = int(os.environ.get("STD_RECORDS","10"))
STD_ATTRS        = int(os.environ.get("STD_ATTRS","5"))

def _worker(args):
    idx, task, max_records, max_attrs, classes, base_path = args
    # folder       = os.path.join(base_path)
    dataset_path = os.path.join(base_path, "datasets")
    labels_path  = os.path.join(base_path, "labels")

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(labels_path,  exist_ok=True)

    rng = default_rng()  # un único seed
    n = min(int(abs( rng.normal(loc=max_records, scale=STD_RECORDS)))+1, max_records)
    max_ops = 10
    i =0
    while n <= 1 or i >= max_ops: 
        n = min(int(abs( rng.normal(loc=max_records, scale=STD_RECORDS)))+1, max_records)
        i+=1
    i=0

    d = min(int(abs(rng.normal(loc=max_attrs,  scale=STD_ATTRS )))+1, max_attrs)
    # print(idx,n,d)

    k = None
    if task == "CLUSTERING":
        k = k_generator(loc=classes, scale=1)
        while k > n:
            print(k, n)
            k = k_generator(loc=classes, scale=1)
            # k = max(2,_k)
            # k = int(_k) if _k >= 2 else int(2)

    ds_id    = f"{idx:07d}{task.lower()}_r{n}_a{d}" + (f"_k{k}" if k else "")
    file_ds  = os.path.join(dataset_path,  f"{ds_id}.npy")
    file_lbl = os.path.join(labels_path,  f"{ds_id}_labels.npy")

    if task == "CLUSTERING":
        dataset, target = data_generator_clustering(n, d, k)
        np.save(file_ds, dataset)
        np.save(file_lbl, target)
        algs = ["KMEANS","SKMEANS","DBSKMEANS","SKMEANSPQC","DBSKMEANSPQC","NNC","DBSNNC"]

        trace_chunk = trace_generator(
            experiment_id    = idx,
            task_id          = task,
            algorithms       = algs,
            dataset_id       = ds_id,
            dataset_filename = ds_id,
            k                = k,
            avg_records      = max_records,
            avg_attributes   = max_attrs
        )

    elif task == "CLASSIFICATION":
        modelFilename       = "{}model".format(ds_id)
        modelLabelsFilename = "{}modellabels".format(ds_id)
        dataFilename        = "{}data".format(ds_id)
        dataLabelsFilename  = "{}datalabels".format(ds_id)

        file_ds           = "{}/{}.npy".format(dataset_path,modelFilename)
        modelLabelsPath     = "{}/{}.npy".format(dataset_path,modelLabelsFilename)
        dataPath            = "{}/{}.npy".format(dataset_path,dataFilename)
        dataLabelsPath      = "{}/{}.npy".format(labels_path,dataLabelsFilename)
        
        model, data, model_labels, data_labels = data_generator_classification(n, d, classes,np.random.randint(low=0, high=100))
        # np.save(file_ds, data)
        # np.save(file_lbl, data_labels)
        np.save(file_ds, model) #Guarda modelo
        np.save(dataPath, data)   #Guarda dataset
        np.save(modelLabelsPath, model_labels) #Guarda etiquetas del modelo
        np.save(dataLabelsPath, data_labels)   #Guarda etiquetas del dataset

        algs = ["KNN","SKNN","SKNNPQC"]
        trace_chunk = trace_generator(
            experiment_id         = idx,
            task_id               = task,
            algorithms            = algs,
            model_id              = ds_id,
            model_filename        = modelFilename,
            model_labels_filename = modelLabelsFilename,
            record_test_id        = dataFilename,
            record_test_filename  = dataFilename,
            avg_records           = max_records,
            avg_attributes        = max_attrs
        )
    size_ds = os.path.getsize(file_ds)
    return idx, size_ds, trace_chunk, algs

# — Función principal con paralelismo —
def generate_batch_datasets(total_size_mb: float, max_records: int, max_attrs: int, classes: int, base_path: str, task: str = "CLUSTERING", max_workers: int = 2):
    target_bytes  = int(total_size_mb * 1024**2)
    traces_path  = os.path.join(base_path, "traces")
    os.makedirs(traces_path,  exist_ok=True)

    trace_columns = [
        "EXPERIMENT_ID","TASK_ID","ALGORITHM","DATASET_ID","DATASET_FILENAME",
        "MODEL_ID","MODEL_FILENAME","MODEL_LABELS_FILENAME","RECORD_TEST_ID",
        "RECORD_TEST_FILENAME","K","MAX_ITERATIONS","SENS","THRESHOLD",
        "INTERARRIVAL_TIME","EXTENSION"]
    global_traces = {
        "CLUSTERING":   {"KMEANS":[], "SKMEANS":[], "DBSKMEANS":[],
                         "SKMEANSPQC":[], "DBSKMEANSPQC":[], "NNC":[], "DBSNNC":[]},
        "CLASSIFICATION":{"KNN":[], "SKNN":[], "SKNNPQC":[]}
    }
    traces      = global_traces[task]
    total_bytes = 0
    pbar        = tqdm(total=target_bytes, unit='B', unit_scale=True, desc="Generando datasets")

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = {}
        idx = 0

        while total_bytes < target_bytes:
            args = (idx, task, max_records, max_attrs, classes, base_path)
            fut = exe.submit(_worker, args)
            futures[fut] = idx
            idx += 1

            # recogemos cualquier tarea terminada
            done, _ = as_completed(futures), None
            for fut in done:
                i, size_ds, trace_chunk, algs = fut.result()
                total_bytes += size_ds
                pbar.update(size_ds)
                for j, row in enumerate(trace_chunk):
                    traces[algs[j]].append(row)
                del futures[fut]
                if total_bytes >= target_bytes:
                    break
        pbar.close()

    # volcamos los CSV finales
    folder = os.path.join(traces_path)
    for alg, rows in traces.items():
        pd.DataFrame(rows, columns=trace_columns) \
          .to_csv(os.path.join(folder, f"{alg}.csv"), index=False)

    print(f"\nGenerados {idx} datasets en '{os.path.join(base_path)}'")
    print(f"Tamaño acumulado ≈ {total_bytes/1024**2:.2f} MB.")

def k_generator(loc:int=0, scale:int=1):
    # while True:
    x = np.ceil(abs(stats.norm.rvs(loc = loc, scale = scale)))
    # if x >= 2:
    # return int(x) if x >= 2 else int(2)
    return max(2, int(x))
    # return int(x)

def interarrivalTime_generator():
    N = 1
    mean_arrival_rate      = 0.5
    mean_interarrival_time = 1/ mean_arrival_rate
    interarrival_times     = np.random.exponential(scale=mean_interarrival_time, size=N)
    return int(np.ceil(interarrival_times[0]))

def data_generator_clustering(records:int, attributes:int, centers:int):    
    features, target = make_blobs(
        n_samples   = records, # Cantidad de filas
        n_features  = attributes,  # Cantidad de columnas
        centers     = centers, # k 
        cluster_std = 0.02, # Modifica la separacion intra-cluster
        shuffle     = True, # Componnente aleatorio
        center_box  = (-1,1)
    )
    return features, target

def data_generator_classification(records:int, attributes:int, classes:int,random_state:int=42):
    clusters_per_class = 1
    required_info      = math.ceil(math.log2(classes * clusters_per_class))
    # Para tener al menos required_info variables informativas,
    # necesitamos attributes ≥ required_info + 1
    min_attrs = required_info + 1

    if attributes < min_attrs:
        attributes = min_attrs

    max_info      = attributes - 1
    # Ahora el n_informative puede elegirse como antes, pero
    # siempre estará ≥ required_info
    n_informative = min(max_info, 2)
    n_informative = min(max_info, max(required_info, 1))
    X, y = make_classification(
        n_samples            = records, 
        n_features           = attributes, 
        n_informative        = n_informative, # 2 
        n_redundant          = 0, 
        n_clusters_per_class = 1, 
        n_classes            = classes, 
        flip_y               = 0.01, 
        class_sep            = 1.5, 
        random_state         = random_state
    )

    # 2) Calcular tamaños garantizando al menos 1 en cada conjunto
    # n_test  = max(1, int(records * 0.2))
    # n_train = records - n_test
    # if n_train < 1:
    #     n_train = 1
    #     n_test  = records - 1

    # idxs       = np.random.permutation(records)
    # train_idxs = idxs[:n_train]
    # test_idxs  = idxs[n_train:n_train + n_test]

    # X_train = X[train_idxs]
    # y_train = y[train_idxs]
    # X_test  = X[test_idxs]
    # y_test  = y[test_idxs]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 

def trace_generator(experiment_id:str, task_id:str, algorithms:str, k:int="", num_chunks:str="", dataset_id:str="", dataset_filename:str="", 
                    model_id:str="", model_filename:str="", model_labels_filename:str="", record_test_id:str="", record_test_filename:str="", avg_records:int=10, avg_attributes:int=1):
    extension         = DATASET_EXTENSION
    threshold         = THRESHOLD
    sens              = SENS
    max_iterations    = MAX_ITERATIONS
    interarrival_time = interarrivalTime_generator()
    # num_chunks        = NUM_CHUNKS
    
    rows = []
    for algorithm in algorithms:
        did = f"{dataset_id}_{algorithm}_{avg_records}_{avg_attributes}".lower()
        mid = f"{model_id}_{algorithm}_{avg_records}_{avg_attributes}".lower()
        row = [experiment_id, task_id, algorithm, did, dataset_filename, mid, model_filename, model_labels_filename, record_test_id, 
               record_test_filename, k, max_iterations, sens, threshold, interarrival_time, extension]
        rows.append(row)
    return rows


if __name__ == "__main__":
    generate_batch_datasets(
    total_size_mb = EXPECTED_SIZE_MB,
    max_records   = MAX_RECORDS,
    max_attrs     = MAX_ATTRIBUTES,
    classes       = CLASSES,
    task          = TASK_ID,
    base_path     = BASE_PATH,
    max_workers   = MAX_PROCESS,
)
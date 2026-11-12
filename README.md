# Machine learning contest GDG on campus UAM 🚀🏆
### 1st Edition

- 🔗 [Competition](https://gdguam.es/blog/guia-concurso-de-ml)
- 🔗 [Repo with contest materials](https://github.com/GDG-UAM/ml-contest-materials.git)


## Approach

### General architecture design:
- **Composable modules:** The project is designed following a modular structure (preprocessing, feature extraction, classifier). These components are decoupled and orchestrated inside a unified, end-to-end time series pipeline (defined in src/training/pipeline.py). This pipeline is built to handle both batch fitting and single-sample inference at runtime.
- **Local testing:** Each component is locally tested under test/ using pytest. This guarantees that feature extractions, preprocessing filters, and incremental steps are correct and do not leak state.
- **Exploratory analysis:** Basic data exploration, prototyping of feature extractors, and proof-of-concept tests are documented in notebooks within the notebooks/ directory.
- **"Template" approach:** The engineering design decouples hyperparameters and recipes. Different configurations are specified via YAML files in configs/, allowing us to experiment with new feature counts or classifier parameters without modifying the training engine.
- **Decoupled low-latency inference:** The entrypoint model.py is generic. It implements the competition contract and loads the latest serialized pipeline artifact (artifacts/latest.joblib) unless overridden by the MODEL_WEIGHTS environment variable.
- **Training workflow:** The script src/train.py orchestrates the experiment lifecycle:
  1. Parses a YAML config file.
  2. Runs optional stratified cross-validation generating out-of-fold (OOF) predictions to evaluate performance without leakage.
  3. Fits the complete time series pipeline on the target dataset.
  4. Exports the final model artifact to a timestamped parameter file, updating also the symbolic latest references in artifacts/.
  5. Computes and saves evaluation metrics (accuracies, abstention rate, confusion matrix) in a timestamped .json under artifacts/metrics/.

---

## Proposal

### Configuration overview and justification:
We implement a pipeline configured in configs/minirocket.yaml under the name minirocket2500-full. It relies on signal preprocessing, random kernel feature maps, and linear classification:

- **Preprocessing:** 
  - **NaN interpolation:** Cleans missing observations in the channels to avoid propagation issues.
  - **Hampel filter (window: 5, n_sigmas: 3.0):** Applied channel-wise to detect and filter out high-amplitude outliers or spikes without distorting the underlying signal structure.
- **Feature extractor:** 
  - We concatenate standard signal descriptors (baseline), summary stats, and a custom implementation of MiniRocket (minirocket) generating 2500 random features. 
  - All extracted features are globally standardized (normalize_features: true) prior to classification.
- **Classifier:** 
  - **Ridge classifier:** We train a scikit-learn RidgeClassifierCV configured with balanced class weights, optimizing the penalty parameter alpha via internal cross-validation to maximize the F1-Macro score.
  - *Alternative out-of-core learning:* We developed and tested an exact, memory-efficient incremental version (IncrementalRidgeClassifier and SGDLinearClassifier in src/classifiers/classifiers.py) capable of training on massive sets via Gram matrix accumulation or Stochastic Gradient Descent. While functional, it was not necessary for the final 100k full-batch training configuration.
- **Abstain logic (The "Noise gate"):**
  - Instead of relying on a margin prediction threshold, we introduce a high-frequency noise gate inside our time series pipeline. 
  - During single-sample inference, we calculate the standard deviation of first-order differences on the clean signal across all channels: max(std(diff(x))). If this high-frequency noise metric exceeds noise_threshold: 3.5, the model proactively abstains (-1) to avoid making predictions on low-quality signals, protecting the official geometric mean score.
- **Training dataset strategy:**
  - A stratified validation split of train_100k was initially used for model validation and architecture tuning.
  - The final model submission was refitted using the full available dataset (train_100k and test_demo) to leverage maximum sample diversity.

---

### Performance on available datasets

`minirocket2500-train`

Performance for val dataset (random subset of train_100k):

```bash
[model] run 0  score=26.921  n=19118  acc=0.845  latency=3.1ms  end=time_budget  ok
[model] run 1  score=25.034  n=16624  acc=0.845  latency=3.6ms  end=time_budget  ok
[model] run 2  score=21.490  n=11925  acc=0.847  latency=5.0ms  end=time_budget  ok
rank  project   mean   best  worst  median  trimmed_mean  mean_minus_std  failed_runs  mean_accuracy_all  mean_n_scored  mean_latency_s
---------------------------------------------------------------------------------------------------------------------------------------
    1  model    24.48  26.92  21.49   25.03         24.48           22.23            0             0.8458      1.589e+04        0.003926
```

Performance for test_demo:

```bash
[model] run 0  score=1.168  n=400  acc=0.640  latency=3.3ms  end=stream_exhausted  ok
[model] run 1  score=1.168  n=400  acc=0.640  latency=2.5ms  end=stream_exhausted  ok
[model] run 2  score=1.168  n=400  acc=0.640  latency=2.4ms  end=stream_exhausted  ok
rank  project   mean   best  worst  median  trimmed_mean  mean_minus_std  failed_runs  mean_accuracy_all  mean_n_scored  mean_latency_s
---------------------------------------------------------------------------------------------------------------------------------------
    1  model    1.168  1.168  1.168   1.168         1.168           1.168            0               0.64            400        0.002728
```

`minirocket2500-full` (note that the performance is biased because we are evaluating on data used for training. This is only a proxy)

Performance for test_demo:

```bash
    [model] run 0  score=1.244  n=400  acc=0.650  latency=2.9ms  end=stream_exhausted  ok
    [model] run 1  score=1.244  n=400  acc=0.650  latency=2.3ms  end=stream_exhausted  ok
    [model] run 2  score=1.244  n=400  acc=0.650  latency=2.2ms  end=stream_exhausted  ok
    rank  project   mean   best  worst  median  trimmed_mean  mean_minus_std  failed_runs  mean_accuracy_all  mean_n_scored  mean_latency_s
    ---------------------------------------------------------------------------------------------------------------------------------------
       1  model    1.244  1.244  1.244   1.244         1.244           1.244            0               0.65            400        0.002483
```
---

## Reproducibility

### Local

We manage project environments and tasks using uv and just.

#### 1. Repository setup and dependencies
Install dependencies and prepare the project environment:

```bash
uv sync
```

#### 2. Running unit tests
Validate modules locally with your chosen command:

```bash
# If just is not installed
uvx --from rust-just just test

# If just is installed
just test

# To run a specific test suite (e.g. test_features.py)
just test features
```

#### 3. Configuring and training a model
You can customize configurations under configs/myconfig.yaml. To trigger a training run:

```bash
# Syntax: just train <data_filename_without_npz> <config_name_without_yaml>
just train train_100k minirocket
```

#### 4. Evaluating model performance
Evaluate any model artifact on a target dataset:

```bash
# Syntax: just eval <data_filename_without_npz> <path_to_model_without_joblib>
# Defaults to evaluating artifacts/latest on data/test_demo.npz
just eval
```

#### 5. Additional utilities

```bash
# Print model file size on disk
just model-size latest

# Export requirements.txt from uv lockfile
just deps

# Clean python and test caches
just clean
```

### Competition (Google Colab)

First we clone the required repositories and combine the code

```bash
! git clone https://github.com/Diego-Hernandez-Jimenez/gdg-uam-ml-contest.git
! git clone -q https://github.com/GDG-UAM/ml-contest-materials.git ml-contest-materials

! mv ml-contest-materials/competition_api.py gdg-uam-ml-contest/competition_api.py
! mv ml-contest-materials/harness.py gdg-uam-ml-contest/harness.py

%cd gdg-uam-ml-contest/
```

Then we retrieve data

```bash
! curl -L --create-dirs -o data/train_100k.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/train_100k.npz

! curl -L --create-dirs -o data/test_demo.npz https://github.com/GDG-UAM/ml-contest-materials/releases/latest/download/test_demo.npz
```

We install dependencies of the project

```bash
! uv pip install -r requirements.txt
```

Finally, we evaluate the model on the target data (this command will use the latest.joblib model)

```bash
! python harness.py evaluate --model model.py
! python harness.py evaluate --model model.py --source file:data/test_demo.npz
```
---

## Honest disclaimer about AI use

I have relied heavily on generative AI models (mainly Gemini) to develop the project, especially during the discovery phase and for rapid feature implementation. This is due to the fact that I am not an expert in time series and had limited time. However, the architecture and model choices, testing strategy, algorithms design... for better or worse, are entirely mine :)

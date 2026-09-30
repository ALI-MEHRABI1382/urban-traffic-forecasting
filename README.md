# Urban Traffic Forecasting

## Overview

This project evaluates whether environmental noise measurements improve next-hour traffic forecasting. LSTM, GRU and Conv1D models are compared using two input configurations: historical traffic counts alone, and traffic counts combined with noise measurements. A persistence forecast provides a simple reference baseline.

The project originated as individual coursework at Budapest University of Technology and Economics in 2025. The repository contains the forecasting pipeline, an executed analysis notebook, data extracts and experiment results.

## Data

The analysis uses hourly Dublin traffic and environmental noise extracts from May–December 2022. Traffic observations are selected from **SCATS site 755, detector 1**. The coursework report identifies the study area as Ballymun Road and the noise source as Sonitus sensor 4; the sensor locations have not been independently verified from the supplied extracts.

- **Traffic:** `Sum_Volume`, the number of vehicles recorded during the hour ending at the timestamp.
- **Noise:** `LAeq`, equivalent continuous sound level in dB.
- **Overlapping period:** 1 May 2022, 01:00 to 25 December 2022, 13:00.

Source attribution and extract provenance are documented in [data/README.md](data/README.md).

## Methodology

Each forecast uses the previous **nine hours** to predict the traffic count for the following hour. Evaluation follows a rolling one-step procedure: earlier observed test hours may be used as inputs to later predictions. Future traffic and noise observations are excluded from each input window.

Data preparation preserves recorded zero traffic counts and excludes conflicting readings at three duplicate timestamps. Observations are aligned to an hourly grid. Input gaps are forward-filled for at most two hours; windows containing unresolved missing values are excluded. Missing traffic targets are neither filled nor smoothed. Input and target scalers are fitted on training observations only.

The hourly timeline is divided chronologically into **60% training, 20% validation and 20% testing**. Validation begins on 21 September 2022, 04:00, and testing begins on 7 November 2022, 21:00. Both input configurations use identical eligible target timestamps: **3,423 training, 1,142 validation and 1,145 test samples**.

### Model configurations

| Model | Main layers | Dropout after each main layer |
| --- | --- | --- |
| Traffic-only LSTM | 64 → 32 units | 40% |
| Traffic + noise LSTM | 64 → 32 units | 30% |
| GRU, both configurations | 64 → 32 units | 30% |
| Conv1D, both configurations | 64 → 32 filters; kernel size 2; ReLU; same padding | 30% |

All architectures use a Dense 8 layer with ReLU followed by a linear Dense 1 output. Conv1D includes a Flatten layer before the dense layers.

Training uses Adam with an initial learning rate of 0.001, mean squared error loss, batch size 32 and a maximum of 50 epochs. Early stopping monitors validation loss with patience 5 and restores the best weights. The learning rate is halved after three epochs without validation improvement. A fixed seed of 42 is used.

The LSTM configurations have different dropout rates; their comparison changes both the feature inputs and regularization. GRU and Conv1D use matching architectures and dropout across the input configurations.

## Results

Models are selected using **validation RMSE**. Test MAE and RMSE are calculated after converting predictions back to hourly vehicle-count units. Lower values indicate smaller errors.

| Model | Inputs | Epochs run | Validation RMSE | Test MAE | Test RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Persistence | Baseline | 0 | 24.14 | 12.76 | 19.48 |
| LSTM | Traffic only | 49 | 19.77 | 11.48 | 16.54 |
| GRU | Traffic only | 49 | 19.14 | 10.61 | 15.45 |
| Conv1D | Traffic only | 35 | 18.24 | 10.18 | 14.95 |
| LSTM | Traffic + noise | 45 | 19.50 | 11.30 | 16.38 |
| GRU | Traffic + noise | 28 | 19.66 | 10.99 | 16.11 |
| Conv1D | Traffic + noise | 25 | 18.09 | 10.32 | 15.22 |

The validation-selected model was **Conv1D with traffic and noise inputs**, with test RMSE **15.22**, compared with **19.48** for persistence: a **21.9% reduction** in this run.

Traffic-only Conv1D achieved a lower test RMSE of **14.95**, although it was not the validation-selected configuration. Adding noise reduced LSTM test error slightly but increased GRU and Conv1D test error. These results do not demonstrate a consistent forecasting benefit from noise measurements. The LSTM dropout difference also limits attribution of its result to the additional feature.

![Test RMSE comparison](figures/model_comparison.png)

![Observed and predicted test traffic](figures/test_predictions.png)

These results use raw traffic targets and should not be directly compared with scores from an earlier experiment using smoothed targets. Full metrics, predictions, training histories and run configuration are stored in `results/`.

## Limitations

The experiment covers one traffic detector, one held-out period and one random seed. The supplied CSVs are processed coursework extracts with incomplete earlier preprocessing records. Timestamps contain no explicit timezone information, so daylight-saving alignment is unverified. Detector faults and short forward fills may affect results; the two-hour filling limit is an assumption rather than a demonstrated optimum.

Further evaluation could include seasonal baselines, repeated seeds, temporal cross-validation and additional locations. Model development should continue to use validation data rather than repeatedly tuning against the reported test results.

## Reproducing the experiment

**Requirements:** Python 3.11 or 3.12. Core package versions are pinned in `requirements.txt`; the included run used Python 3.12.14 and TensorFlow CPU 2.20.0. A GPU is not required.

After cloning or downloading and extracting the repository, run the following commands from its root folder:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `traffic_forecasting.ipynb` using that Python environment and select **Restart Kernel and Run All Cells**. The default configuration is `RETRAIN = True`, `MAX_EPOCHS = 50` and `BATCH_SIZE = 32`. The notebook contains the full pipeline code and reports training progress. Dependencies can alternatively be installed into the active notebook kernel using the commented `%pip install -r requirements.txt` cell, followed by a kernel restart.

The equivalent command-line experiment is:

```bash
python forecasting.py --epochs 50 --batch-size 32
```

Training regenerates `results/`, `figures/` and `models/`. Setting `RETRAIN = False` displays the committed results without training; the checkpoint verification requires locally generated model files. Runtime and small numerical differences may vary across environments.

## Repository structure

| Path | Contents |
| --- | --- |
| `traffic_forecasting.ipynb` | Full analysis code and saved outputs |
| `forecasting.py` | Equivalent command-line pipeline |
| `requirements.txt` | Dependency versions |
| `data/` | CSV extracts and source attribution |
| `results/` | Metrics, predictions, histories and run details |
| `figures/` | Data, model comparison, prediction and training charts |
| `models/` | Generated locally: checkpoints and fitted scalers; excluded from Git |

Saved checkpoints expect scaled inputs and produce scaled outputs. The matching preprocessing objects are generated in `models/preprocessing.joblib`.

## Author

**Ali Mehrabi** — Computer Science student, Budapest University of Technology and Economics.  
[GitHub profile](https://github.com/ALI-MEHRABI1382)

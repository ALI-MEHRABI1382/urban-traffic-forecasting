# Agreed changes from the original coursework notebook

## Restored exactly from the original architecture definitions

| Model | Main layers | Dropout |
| --- | --- | --- |
| Traffic-only LSTM | LSTM 64 → LSTM 32 | 0.4 after each LSTM |
| Traffic + noise LSTM | LSTM 64 → LSTM 32 | 0.3 after each LSTM |
| Both GRU variants | GRU 64 → GRU 32 | 0.3 after each GRU |
| Both Conv1D variants | 64 → 32 filters, kernel size 2, ReLU, same padding | 0.3 after each convolution |

All models retain Dense 8 with ReLU and a linear Dense 1 output. Conv1D retains Flatten before the dense layers. The first recurrent layer returns sequences; the second does not.

The original LSTM variants used different dropout rates. This difference is preserved, and means that their comparison changes both dropout and feature inputs. GRU and Conv1D use matching architectures across feature variants.

## Restored training settings

- Maximum 50 epochs and explicit batch size 32 (the original NumPy-based fit default).
- Adam, initial learning rate 0.001, MSE loss and RMSE tracking.
- EarlyStopping: validation loss, patience 5, restore best weights.
- ReduceLROnPlateau: validation loss, patience 3, factor 0.5.
- ModelCheckpoint: save the best validation-loss model for each variant.

Training uses a fixed seed and explicit dataset shuffling. Exact historical numerical results are not expected: preprocessing, scaling and the evaluation targets have changed.

## Agreed data corrections retained

- Portable paths; explicit numerical and timestamp conversion.
- Explicit site 755 and detector 1 selection (the supplied file only contains site 755).
- Preserve recorded zero traffic counts.
- Exclude both conflicting readings at each of three duplicate timestamps.
- Reindex onto an hourly grid before constructing nine-hour windows.
- Forward-fill input features for at most two hours; skip windows still incomplete.
- Never fill missing evaluation targets.
- Remove exponential smoothing from noise and traffic; predict raw recorded next-hour traffic counts.
- Establish chronological 60/20/20 date boundaries and use the same eligible targets for every variant.
- Fit scaling using training data only, actually train on scaled arrays, and convert predictions back before reporting errors.

The two-hour filling limit is a chosen assumption, not a demonstrated optimum. The noise <= 0 exclusion is also an assumption, with no effect on this supplied noise file, which contains no such readings.

## Evaluation and reproducibility additions

- Persistence baseline using the latest available traffic feature.
- Select by validation RMSE; report test MAE and RMSE for all variants.
- Save metrics, predictions, data quality counts, training histories, package versions, all six model checkpoints and fitted scalers.
- Include all pipeline code directly in the notebook and an equivalent command-line script.
- The notebook trains by default; saved outputs let visitors preview the completed run.

The earlier 30-epoch/smaller-model package is superseded. Its scores and percentage improvements do not describe this revision.

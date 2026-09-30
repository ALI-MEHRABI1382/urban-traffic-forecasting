# Run this project in Jupyter

1. Extract the **entire ZIP** into a new folder. Do not mix these files with the older package.
2. Open `traffic_forecasting.ipynb` from inside the extracted `urban-traffic-forecasting` folder. Keep `data/` beside the notebook.
3. Select a **Python 3.11 or 3.12** kernel. The full verification used Python 3.12.
4. If this kernel needs dependencies, uncomment the `%pip install -r requirements.txt` line near the top, run that cell once, and restart the kernel. `%pip` installs into the notebook's current environment.
5. Keep `RETRAIN = True`, `MAX_EPOCHS = 50`, and `BATCH_SIZE = 32`.
6. Choose **Restart Kernel and Run All Cells**. Let all six models finish; progress appears in the training cell.
7. The last code cell checks that a saved model reloads and reproduces its reported predictions.

Outputs are refreshed in `results/`, `figures/` and `models/`. The model files are included for your local use and excluded by `.gitignore`. Scaled training RMSE printed during epochs is different from the final RMSE in vehicle counts; use the final metrics table for interpretation.

If Python is missing or your kernel is a different version, create a Python 3.12 environment first. In Windows Command Prompt, from this folder:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

On macOS/Linux with Python 3.12 installed:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Core packages are pinned to the tested versions to reduce environment differences. Local hardware and operating system differences can still change runtime and numerical results.

# Optimizing Solar Tracking using AI

A study applying neural networks to predict distributed solar energy output from sun position data. Built using PyTorch and real county-level generation data from the Lawrence Berkeley National Laboratory.

---

## Overview

This project trains a feedforward Artificial Neural Network (ANN) to predict hourly solar energy output (MWh) for Gilmer County, Georgia, using computed solar altitude and azimuth as input features. The goal is to establish a baseline model that could inform real-time solar panel orientation and energy forecasting systems.

---

## Project Structure

```
.
├── Copy_of_Optimizing_Solar_Tracking_using_AI.ipynb  # Full experiment and analysis
├── ann_solar_model.pth                               # Saved ANN model weights
├── requirements.txt                                  # Python dependencies
├── DECISIONS.md                                      # Design decisions and motivation
├── TODO.md                                           # To-do list: Planned Updates 
└── README.md
```

> **Note:** The raw dataset (`13123.csv`) is not included in this repository due to its size. See the Data section below for instructions on obtaining it.

---

## Data

County-level hourly distributed PV generation data sourced from the Lawrence Berkeley National Laboratory Solar-to-Grid Public Data File (2012–2020).

- **Source:** [OEDI — Solar-to-Grid Public Data File](https://data.openei.org/submissions/4503)
- **DOI:** [10.25984/1825661](https://doi.org/10.25984/1825661)
- **File used:** `13123.csv` — FIPS code for Gilmer County, Georgia (FIPS: 13-123)
- **Time range:** May 1, 2019 – October 1, 2019 (daytime hours only)

To reproduce this experiment, download the `Hourly Generation by Plant and County.zip` file from the link above and place `13123.csv` in the project root.

---

## Methodology

**1. Data Pruning**
Raw hourly generation data is filtered to the peak solar season (May–September 2019). Residential and non-residential distributed PV values are combined into a single `DPV_Combined_MWh` column. Nighttime rows where solar altitude ≤ 0 are excluded, reducing the dataset from 3,672 to 2,070 daytime samples.

**2. Feature Engineering**
Solar altitude and azimuth are computed per timestamp using `pysolar` at the coordinates for Gilmer County, GA (34.77°N, -84.57°W). Azimuth is encoded as sine and cosine components to avoid the 0°/360° angular discontinuity.

**3. Model Architecture**
A three-layer feedforward ANN implemented in PyTorch:
- Input layer: 3 features (Solar Altitude, Azimuth Sin, Azimuth Cos)
- Hidden layers: 2 × 32 neurons with ReLU activation
- Output layer: 1 neuron (predicted MWh)

**4. Training**
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (lr = 0.0001)
- Scheduler: ReduceLROnPlateau (patience = 10, factor = 0.5)
- Batch size: 64
- Epochs: 100
- Split: 70% train / 15% validation / 15% test

**5. Evaluation**
Performance measured using MAE and RMSE on the held-out test set.

---

## Results

| Model | Location | Data Period | Test MAE | Test RMSE | Date Run |
|-------|----------|-------------|----------|-----------|----------|
| ANN | Gilmer County, GA | May–Sep 2019 | 0.005151 MWh | 0.007584 MWh | Mar 6, 2026 |

On a target range of 0–0.047 MWh, the ANN achieves approximately 11% mean absolute error, converging smoothly with no signs of overfitting.

---

## Setup

**Requirements**
- Python 3.11

**Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

**Run**

Open the notebook directly:
```bash
jupyter notebook Copy_of_Optimizing_Solar_Tracking_using_AI.ipynb
```

To load the saved model weights without retraining:
```python
model = SolarTrackingNN()
model.load_state_dict(torch.load('ann_solar_model.pth'))
model.eval()
```

---

## References

Seel, J., Mills, A., Millstein, D., Gorman, W., & Jeong, S. (2021). *Solar-to-Grid Public Data File for Utility-scale (UPV) and Distributed Photovoltaics (DPV) Generation, Capacity Credit, and Value for 2012-2020.* Lawrence Berkeley National Lab. https://doi.org/10.25984/1825661
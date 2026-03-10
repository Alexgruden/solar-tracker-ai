# Design Decisions

This document records the key technical decisions made during development of the Solar Tracking ANN, including the reasoning behind each choice. It exists so that the project's logic is transparent and reproducible.

---

## Data Scope

**Decision:** Filter to daytime hours only (`Solar_Altitude > 0`), reducing the dataset from 3,672 to 2,070 samples.

**Reasoning:** Nighttime rows always produce zero energy output by definition. Including them would add trivial samples that the model could exploit without learning anything meaningful about the solar-energy relationship. Training on daytime-only data keeps the model focused on the pattern that actually matters.

---

## Feature Engineering — Azimuth Encoding

**Decision:** Encode solar azimuth as `sin(azimuth)` and `cos(azimuth)` rather than using the raw degree value.

**Reasoning:** Raw azimuth in degrees has a discontinuity at the 0°/360° boundary — two physically adjacent sun positions (e.g., 359° and 1°) appear numerically far apart. A neural network would have to learn to ignore this artifact. Sin/cos encoding wraps azimuth onto a unit circle, making the representation continuous and removing the discontinuity entirely.

---

## Feature Selection

**Decision:** Use three input features: Solar Altitude, Azimuth Sin, Azimuth Cos. Exclude timestamp and installed capacity columns.

**Reasoning:** Solar altitude and azimuth are the physical quantities that govern how much sunlight reaches the panels. Timestamps are a proxy for sun position, not the cause of generation — using them directly would be a form of data leakage that wouldn't generalise to different years or locations. `DPV_Combined_MW` represents slowly-incrementing installed capacity, not actual hourly generation, so it was excluded to avoid training on the wrong signal.

---

## Target Variable

**Decision:** Predict `DPV_Combined_MWh` (actual energy generated), not `DPV_Combined_MW` (installed capacity).

**Reasoning:** MW in this dataset increments slowly over years as new panels are installed — it is not a measure of hourly output. MWh is the actual generation value that varies with sun position and is the correct regression target.

---

## StandardScaler — Fit on Training Data Only

**Decision:** Fit the `StandardScaler` on `X_train` only, then apply the same transformation to validation and test sets.

**Reasoning:** Fitting the scaler on the full dataset would leak information about the distribution of the validation and test sets into training. Fitting on training data only ensures that the model never sees statistical properties of the held-out data during training, preserving the integrity of the evaluation.

---

## Loss Function — MSELoss for Training

**Decision:** Use Mean Squared Error (MSE) as the training loss function. Report MAE and RMSE at evaluation time.

**Reasoning:** MSE penalises larger errors more heavily than MAE, which is appropriate for a regression task where large prediction errors are more costly than small ones. MAE and RMSE are reported at test time because they are more interpretable in the original units (MWh) and easier to communicate.

---

## Optimizer and Learning Rate

**Decision:** Adam optimizer with learning rate `0.0001`.

**Reasoning:** Adam adapts the learning rate per parameter and handles sparse gradients well, making it a robust default for feedforward networks. A learning rate of 0.0001 was chosen after observing unstable loss curves at higher rates during early runs.

---

## Learning Rate Scheduler — ReduceLROnPlateau

**Decision:** Apply `ReduceLROnPlateau` with `patience=10` and `factor=0.5`, stepping on validation loss.

**Reasoning:** Early training runs produced noisy, non-converging loss curves. The scheduler reduces the learning rate by half whenever validation loss stops improving for 10 consecutive epochs, allowing the model to make large updates early in training and fine-tune as it approaches convergence. This produced smooth, stable loss curves within 100 epochs.

---

## Batch Size

**Decision:** Batch size of 64.

**Reasoning:** An initial batch size of 32 produced noisy gradient updates and unstable loss curves. Increasing to 64 stabilised training without requiring a corresponding increase in epochs. Larger batches provide more stable gradient estimates at the cost of some regularisation benefit — 64 was a practical balance for this dataset size.

---

## Architecture — Two Hidden Layers of 32 Neurons

**Decision:** Two hidden layers, each with 32 neurons and ReLU activation.

**Reasoning:** The task is a low-dimensional regression problem with 3 inputs and 1 output. A small network is appropriate — a large one would risk overfitting on 2,070 samples. Two layers of 32 neurons is sufficient capacity to model the non-linear relationship between sun position and energy output without overcomplicating the architecture.

---

## Train / Validation / Test Split

**Decision:** 70% training, 15% validation, 15% test. Splits performed with `random_state=42` for reproducibility.

**Reasoning:** A three-way split ensures that the learning rate scheduler (which steps on validation loss) does not indirectly influence the test evaluation. The test set is held out entirely until final evaluation, giving an unbiased estimate of generalisation performance.
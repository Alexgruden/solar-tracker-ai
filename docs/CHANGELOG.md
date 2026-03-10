# Changelog

All notable changes and bug fixes made during development are documented here.

---

## [1.0.0] — March 2026

### Fixed

**X/Y variable swap**
The original notebook had timestamps and installed MW capacity as inputs and solar angles as outputs — the inverse of the intended design. Corrected to use solar altitude and azimuth as inputs and `DPV_Combined_MWh` as the regression target.

**Loss averaging divisor**
`running_train_loss` was divided by `len(train_loader)` (number of batches) instead of `len(train_loader.dataset)` (number of samples). This caused the reported training loss to scale with batch count rather than dataset size, making epoch losses incomparable across different batch sizes. Fixed in both training and validation phases.

**Unreachable test block**
The `test_model()` call was indented inside the function definition, below the `return` statement. The test was never actually executing. Fixed by moving the call outside the function scope.

**Incorrect accuracy metric for regression**
The original evaluation used `outputs.round() == targets` — a classification metric that is meaningless for continuous regression outputs. Replaced with MAE and RMSE, which measure prediction error in the original units (MWh).

**MAE divisor error**
`running_val_mae` was divided by `targets.size(1)` — the number of output features (3 at the time) — instead of `len(val_loader.dataset)`, the number of samples. This caused MAE to be reported as 3× higher than the true value.

**RMSE print statement**
The `Test RMSE` print statement was outputting `average_mae` instead of `average_rmse`. Fixed to print the correct variable.

**Nighttime data included in training**
Rows where `Solar_Altitude <= 0` (nighttime) were included in the dataset. These samples always produce zero energy output and add no useful signal, distorting the learned relationship. Fixed by filtering to daytime-only samples, reducing the dataset from 3,672 to 2,070 rows.

**Duplicate imports**
`import pandas as pd` and `import matplotlib.pyplot as plt` each appeared twice in the notebook. Removed duplicate import statements.

**Wrong target column — MW vs MWh**
`DPV_Combined_MW` represents slowly-incrementing installed panel capacity, not hourly energy generation. Using it as the training target meant the model was learning an unrelated signal. Corrected to use `DPV_Combined_MWh`, the actual hourly generation value.
# vect_auto_regr
## Features

- **Vector Autoregression:** Simulate time series data for multivariate systems.
- **Custom Initial Conditions:** Set initial states for the simulation.
- **Bias Adjustment:** Optionally include a bias vector to the model dynamics.
- **Companion Matrix Generation:** Automatically generate a companion matrix which helps in assessing the stability of the vector autoregressive process.
- **Stability Check:** Evaluate the stability of the system using the eigenvalues of the companion matrix. A stable system is indicated when all eigenvalues have absolute values less than 1.

## Usage

To use the `VectAutoReg` class, define the coefficients and optionally the bias and initial state. Then generate the series:

```python
import numpy as np
import matplotlib.pyplot as plt
from vect_autoreg import VectAutoReg

# Define the model coefficients
A1 = np.array([[0.5, 0.1], [0.1, 0.5]])
A2 = np.array([[0.3, 0.0], [0.0, 0.3]])

# Create an instance of the VectAutoReg model
autoreg = VectAutoReg([A1, A2])

# Generate a time series of length 1000
series = autoreg(1000)

# Plot the series
for s in series:
    plt.plot(s)
plt.show()

# Generate and print the companion matrix
companion_matrix = autoreg._generate_companion()
print("Companion Matrix:\n", companion_matrix)

# Check and print if the process is stable
is_stable = autoreg._compute_stability()
print("Is the process stable? ", is_stable)


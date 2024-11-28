
# My Data Science Package

This Python package provides a collection of utilities for data science workflows, including data preprocessing, visualization, and model evaluation.

## Features

### Data Processing
- **summarize_dataframe**: Summarizes the structure and content of a DataFrame.
- **impute_missing_values**: Imputes missing values using KNN and Iterative Imputer.

### Data Visualization
- **histogram_boxplot**: Combines histogram and boxplot for a feature.
- **plot_correlation_heatmap**: Displays a heatmap of correlations.

### Model Evaluation
- **metrics_score**: Calculates classification metrics and displays a confusion matrix.

### Utilities
- **segregate_columns_by_dtype**: Categorizes DataFrame columns by their data types.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd my_data_science_package
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

Import the required module and use its functions:
```python
from my_data_science_package.data_processing import summarize_dataframe
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(summarize_dataframe(df))
```

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them to GitHub:
   ```bash
   git push origin feature-name
   ```
4. Submit a pull request.

## License
This project is licensed under the MIT License.

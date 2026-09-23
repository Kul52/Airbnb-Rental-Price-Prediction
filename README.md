# Airbnb Rental Price Prediction

A machine learning project that predicts the nightly price of New York City Airbnb listings from location, room type, availability, and review activity. It is built as an end-to-end scikit-learn `Pipeline`, so all preprocessing and modelling run as a single reproducible object.

## Dataset

The data (`/kaggle/input/datasets/dgomonov/new-york-city-airbnb-open-data/AB_NYC_2019.csv`) contains **27,379 NYC Airbnb listings** with 16 columns:

| Column | Description |
|---|---|
| `id`, `name`, `host_id`, `host_name` | Listing and host identifiers (dropped before modelling) |
| `neighbourhood_group` | Borough: Manhattan, Brooklyn, Queens, Bronx, Staten Island |
| `neighbourhood` | Specific neighbourhood (e.g. Clinton Hill, Chelsea) |
| `latitude`, `longitude` | Listing coordinates |
| `room_type` | Entire home/apt, Private room, or Shared room |
| `price` | **Target**: nightly price in USD |
| `minimum_nights` | Minimum stay length |
| `number_of_reviews` | Total reviews |
| `last_review` | Date of the most recent review |
| `reviews_per_month` | Average monthly reviews |
| `calculated_host_listings_count` | Number of listings the host has |
| `availability_365` | Days per year the listing is available |

Room types are split roughly 52% entire homes, 46% private rooms, and 2% shared rooms. About 20% of listings have no reviews, so `last_review` and `reviews_per_month` contain missing values.

## Approach

### 1. Exploration
Initial exploration checked data types, missing values, and category distributions, and tried manual encoding (one-hot via `pd.get_dummies`, ordinal encoding, and date-to-integer conversion) before moving everything into a pipeline.

### 2. Preprocessing pipeline
A `ColumnTransformer` handles each feature type separately:

| Feature group | Columns | Steps |
|---|---|---|
| One-hot categorical | `room_type`, `neighbourhood_group` | Most-frequent imputation, then `OneHotEncoder` (ignores unseen categories) |
| High-cardinality categorical | `neighbourhood` | Most-frequent imputation, then `OrdinalEncoder` (unseen categories become -1) |
| Date | `last_review` | Custom `FunctionTransformer` converting dates to days since 2000-01-01, then median imputation |
| Numeric (remainder) | All other columns | Mean imputation, then `StandardScaler` |

Handling imputation inside the pipeline means missing values are filled using training data only, avoiding leakage into the test set.

### 3. Model
The preprocessor feeds into a `LinearRegression` baseline. Data is split 70/30 into training and test sets.

## Results

<!-- Replace with your final numbers -->
| Metric | Value |
|---|---|
| R² (test) | _TBD_ |
| RMSE (test) | _TBD_ |

Linear regression serves as a baseline. Airbnb prices are heavily right-skewed with extreme outliers, which limits how well a linear model can fit them.

## Project Structure

```
├── /kaggle/input/datasets/dgomonov/new-york-city-airbnb-open-data/AB_NYC_2019.csv                   # Dataset
├── airbnb_price_prediction.ipynb  # Notebook with exploration, pipeline and model
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook airbnb_price_prediction.ipynb
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter

## Future Improvements

- Log-transform `price` and remove extreme outliers to handle skew
- Try non-linear models such as Random Forest, Gradient Boosting (XGBoost/LightGBM), and SVR
- Replace ordinal encoding of `neighbourhood` with target encoding, since ordinal numbers imply an order that doesn't exist
- Engineer features like distance to Midtown Manhattan
- Tune hyperparameters with cross-validation (`GridSearchCV`)
- Add visualisations of price distribution by borough and room type

## Author

**Kuldeep Singh Naga**

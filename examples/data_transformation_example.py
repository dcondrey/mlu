from mlu.modules.data_transformation import handle_missing_values, normalize, encode_categorical

import pandas as pd
import numpy as np

def main():
    try:
        # Example DataFrame with missing values
        df = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4],
            'Feature2': ['A', 'B', 'B', 'A']
        })

        # Handling missing values
        df_filled = handle_missing_values(df[['Feature1']], strategy='mean')
        print("DataFrame after handling missing values:\n", df_filled)

        # Normalizing numerical data
        normalized_array = normalize(np.array([1, 2, 3, 4, 5]))
        print("\nNormalized array:\n", normalized_array)

        # Encoding categorical data
        encoded_df = encode_categorical(df[['Feature2']], encoding_type='onehot')
        print("\nDataFrame after encoding categorical data:\n", encoded_df)

    except Exception as e:
        print("An error occurred during data transformation example execution.", e)

if __name__ == "__main__":
    main()
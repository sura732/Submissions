import pandas as pd
import numpy as np

def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Create an empty DataFrame to store the distance matrix
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)

    
    distance_matrix = distance_matrix.fillna(0)

    # Populate the distance matrix with cumulative distances
    for index, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] += distance
        distance_matrix.at[end, start] += distance  # Symmetric

    # Floyd-Warshall Algorithm to find shortest paths
    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if i != k and j != k and distance_matrix.at[i, k] > 0 and distance_matrix.at[k, j] > 0:
                    if distance_matrix.at[i, j] == 0 or distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                        distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]

    # Set diagonal values to 0
    for i in unique_ids:
        distance_matrix.at[i, i] = 0

    return distance_matrix

def unroll_distance_matrix(df):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    for i, row in df.iterrows():
        for j, distance in row.items():
            if i != j and distance > 0:
                unrolled_df = pd.concat([unrolled_df, pd.DataFrame({'id_start': [i], 'id_end': [j], 'distance': [distance]})], ignore_index=True)

    return unrolled_df.astype({'id_start': int, 'id_end': int, 'distance': int})

def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame containing columns 'id_start', 'id_end', and 'distance'.
        reference_id (int): Reference ID for which we calculate the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Filter rows where id_start is equal to the reference_id
    reference_rows = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    reference_average_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds for the threshold
    lower_bound = reference_average_distance - 0.1 * reference_average_distance
    upper_bound = reference_average_distance + 0.1 * reference_average_distance

    # Filter rows where the average distance is within the threshold
    filtered_rows = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]

    # Get unique values from the 'id_start' column and sort them
    result_df = pd.DataFrame(sorted(filtered_rows['id_start'].unique()), columns=['id_start'])

    return result_df


import pandas as pd

def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Multiply the distance by the rate coefficient for each vehicle type
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    # Optionally, you can uncomment the following line to remove the 'distance' columns
    df = df.drop(columns='distance')

    return df









df = pd.read_csv("C:\\Users\\Suraj\\Desktop\\Mapup_assesment\\dataset-3.csv")
result_matrix = calculate_distance_matrix(df)
# Display the resulting distance matrix
print(result_matrix)

# # Call the unroll_distance_matrix function with the distance matrix generated in Question 1
unrolled_result = unroll_distance_matrix(result_matrix)

# Display the resulting unrolled DataFrame
print(unrolled_result)

# Assuming 'unrolled_result' is the DataFrame from Question 2
result = find_ids_within_ten_percentage_threshold(unrolled_result, reference_id=1004356)

# Display the resulting DataFrame
print(result)
# Replace unrolled_result with the actual DataFrame you have
result_with_toll = calculate_toll_rate(unrolled_result)

# Display the resulting DataFrame with toll rates (without 'distance' columns)
print(result_with_toll)

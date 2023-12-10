import pandas as pd

def generate_car_matrix(df):
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns id_1, id_2, car, and others.

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    # Extract relevant columns
    df_car = df[['id_1', 'id_2', 'car']]

    # Pivot the DataFrame
    df = df_car.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for idx in df.index:
        df.loc[idx, idx] = 0

    return df

def get_type_count(df):
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame): Input DataFrame with columns id_1, id_2, route, moto, car, rv, bus, truck.

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    
    # Create a new column 'car_type' based on the specified conditions
    df['car_type'] = pd.cut(df['car'],
                            bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'],
                            include_lowest=True)

    # Count the occurrences of each car type
    type_counts = df['car_type'].value_counts().to_dict()

    return type_counts

def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    
    # Calculate the mean of the 'bus' column
    bus_mean = df['bus'].mean()

    # Filter indexes where 'bus' values are greater than twice the mean
    
    #bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()
    #return sorted(bus_indexes)
    
    bus_indexes = df[df['bus'] > 2 * bus_mean].index
    return sorted(list(bus_indexes ))

def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here

    # Calculate the average 'truck' values for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes with average 'truck' values greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7]

   return sorted(list(selected_routes))



#first function stored in result_matrix ,using result_matrinx and call  the multipry_matrinx
def multiply_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Apply custom conditions
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

from dateutil.relativedelta import relativedelta

def time_check(df):
    # Combine 'startDay' and 'startTime' columns to create a 'start_timestamp' column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    
    # Combine 'endDay' and 'endTime' columns to create an 'end_timestamp' column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Extract day of the week and time components for both start and end timestamps
    df['start_day_of_week'] = df['start_timestamp'].dt.day_name()
    df['start_time'] = df['start_timestamp'].dt.time
    df['end_day_of_week'] = df['end_timestamp'].dt.day_name()
    df['end_time'] = df['end_timestamp'].dt.time
    
    # Calculate the time difference between start and end timestamps using relativedelta
    df['time_difference'] = df.apply(lambda row: relativedelta(row['end_timestamp'], row['start_timestamp']), axis=1)
    
    # Create boolean masks for valid start and end timestamps (covering a full 24-hour period)
    valid_start_time_mask = (df['start_time'] == pd.Timestamp('00:00:00').time()) & \
                            (df['start_timestamp'].dt.date == df['start_timestamp'].dt.date)
    valid_end_time_mask = (df['end_time'] == pd.Timestamp('23:59:59').time()) & \
                          (df['end_timestamp'].dt.date == df['end_timestamp'].dt.date)
    
    # Create boolean masks for valid days of the week (all 7 days present for both start and end timestamps)
    valid_start_day_mask = df['start_day_of_week'].nunique() == 7
    valid_end_day_mask = df['end_day_of_week'].nunique() == 7
    
    # Combine the masks to get the final boolean series
    result_series = df.groupby(['id', 'id_2']).apply(lambda x: 
                            valid_start_time_mask.all() & valid_start_day_mask & 
                            valid_end_time_mask.all() & valid_end_day_mask).astype(bool)
    
    return result_series


df = pd.read_csv("C:\\Users\\Suraj\\Desktop\\Mapup_assesment\\dataset1.csv")
result_matrix= generate_car_matrix(df)
print(result_matrix)
result_matrix_1 = get_type_count(df)
print("get_type_count:",result_matrix_1)
result_matrix_2 = get_bus_indexes(df)
print("get_bus_indexes",result_matrix_2)
modified_result_matrix = multiply_matrix(result_matrix) #Passing the value multiply_matrix
print(modified_result_matrix)

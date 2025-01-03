import pandas as pd
import numpy as np
l = []
for i in range (0, 25):
    l.append(f'../Data/P_Data/Data_{i}.csv')

def reorganize_data(df):
    """
    Reorganize data to group by x-coordinate first, then y-coordinate.
    Input data is assumed to be grouped by y-coordinate first, then x-coordinate.
    """
    # Get the number of unique x and y coordinates
    n_x = len(df['x'].unique())
    n_y = len(df['y'].unique())
    
    # Create a new empty DataFrame with the same columns
    new_df = pd.DataFrame(columns=df.columns)
    
    # Initialize an empty list to store the reordered rows
    reordered_rows = []
    
    # For each x coordinate
    for x in range(n_x):
        # Get all points with this x coordinate, sorted by y
        x_points = df[df['x'] == x].sort_values('y')
        reordered_rows.append(x_points)
    
    # Concatenate all the reordered rows
    new_df = pd.concat(reordered_rows, ignore_index=True)
    
    return new_df

def verify_reorganization(original_df, new_df):
    """
    Verify that the reorganization was successful by checking:
    1. Same number of rows
    2. Same unique values
    3. Pattern of x and y coordinates
    """
    print("Verification Results:")
    print(f"Original shape: {original_df.shape}")
    print(f"New shape: {new_df.shape}")
    print("\nFirst 10 rows of reorganized data:")
    print(new_df[['x', 'y', 'u', 'v', 'p']].head(10))
    print("\nLast 10 rows of reorganized data:")
    print(new_df[['x', 'y', 'u', 'v', 'p']].tail(10))

if __name__ == "__main__":
    # Read the original data
    # Replace with your file path
    for j in range(0, 25):
        file_path = l[j]
        df = pd.read_csv(file_path)
        
        # Reorganize the data
        new_df = reorganize_data(df)
        
        # Verify the reorganization
        verify_reorganization(df, new_df)
        
        # Save the reorganized data
        output_file = f"../Data/P_Data/use/data_new{j}.csv"
        new_df.to_csv(output_file, index=False)
        print(f"\nReorganized data saved to {output_file}")
        
        # Print some example coordinate patterns to verify the ordering
        # print("\nExample coordinate pattern (first 20 points):")
        # print(new_df[['x', 'y']].head(20))
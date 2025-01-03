import pandas as pd
import numpy as np
l = []
for i in range(0, 25):
    l.append(f'../Data/T_Data/newnewnew/Data_{i}.csv')

def add_boundary_column(df, tolerance=1e-6):
    """
    Add a 'Boundary' column to identify stagnation points.
    
    Args:
        df: DataFrame containing u and v velocity components
        tolerance: threshold below which velocity is considered zero
    
    Returns:
        DataFrame with new 'Boundary' column
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_new = df.copy()
    
    # Add Boundary column (1 for stagnation points, 0 otherwise)
    df_new['Boundary'] = ((abs(df_new['u']) < tolerance) & 
                         (abs(df_new['v']) < tolerance)).astype(int)
    
    return df_new

def analyze_boundary_points(df):
    """
    Print analysis of the identified boundary points
    """
    n_boundary = df['Boundary'].sum()
    total_points = len(df)
    
    print("Boundary Points Analysis:")
    print(f"Total points: {total_points}")
    print(f"Number of boundary points: {n_boundary}")
    print(f"Percentage of boundary points: {(n_boundary/total_points)*100:.2f}%")
    
    print("\nSample of boundary points:")
    print(df[df['Boundary'] == 1][['x', 'y', 'u', 'v', 'p', 'Boundary']].head())

if __name__ == "__main__":
    # Read the data
    # Replace with your file path
    for j in range(0, 21):
        file_path = l[j]
        df = pd.read_csv(file_path)
        
        # Add boundary column
        df_with_boundary = add_boundary_column(df)
        
        # Analyze the results
        analyze_boundary_points(df_with_boundary)
        
        # Save the updated DataFrame
        output_file = f"Boundary_Data/data_with_boundary{j}.csv"
        df_with_boundary.to_csv(output_file, index=False)
        print(f"\nUpdated data saved to {output_file}")
        
        # Verify the file was saved correctly
        print("\nVerifying saved file...")
        df_check = pd.read_csv(output_file)
        print("Columns in saved file:", df_check.columns.tolist())
        print("\nSample of saved data:")
        print(df_check[['x', 'y', 'u', 'v', 'Boundary']].head())
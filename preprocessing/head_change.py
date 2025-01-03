import pandas as pd

def rename_columns(input_file, output_file):
    # Read the tab-separated file
    df = pd.read_csv(input_file)
    
    # Define column mapping
    column_map = {
        'U:0': 'u',
        'U:1': 'v',
        'U:2': 'w',
        'vtkValidPointMask': 'boundary',
        'Structured Coordinates:0': 'x',
        'Structured Coordinates:1': 'y',
        'Structured Coordinates:2': 'z'
    }
    
    # Rename columns
    df = df.rename(columns=column_map)
    
    # Save to new location
    df.to_csv(output_file, index=False)


l = []
for i in range (0, 25):
    l.append(f'../Data/T_Data/newnewnew/Data_{i}.csv')
    
m = []
for i in range (0, 25):
    m.append(f'../Data/P_Data/Data_{i}.csv')
    
    
for k in range (0, 25):
    rename_columns(l[k], m[k])
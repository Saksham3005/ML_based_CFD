import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_stagnation_points(df, tolerance=1e-6):
    """
    Find points where both u and v are approximately zero.
    
    Args:
        df: DataFrame containing u, v, x, y coordinates
        tolerance: threshold below which velocity is considered zero
    """
    # Find points where both u and v are close to zero
    stagnation_mask = (abs(df['u']) < tolerance) & (abs(df['v']) < tolerance)
    stagnation_points = df[stagnation_mask]
    
    return stagnation_points

def plot_stagnation_points(df, stagnation_points):
    """
    Create a plot showing all points and highlighting stagnation points.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot all points in light grey
    plt.scatter(df['x'], df['y'], c='lightgrey', alpha=0.5, label='All Points')
    
    # Plot stagnation points in red
    plt.scatter(stagnation_points['x'], stagnation_points['y'], 
               c='red', marker='x', s=100, label='Stagnation Points')
    
    # Add velocity magnitude as color background
    velocity_mag = np.sqrt(df['u']**2 + df['v']**2)
    plt.tricontourf(df['x'], df['y'], velocity_mag, levels=20, alpha=0.3, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude')
    
    # Customize the plot
    plt.title('Stagnation Points (u = v = 0)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make sure axis scales are equal
    plt.axis('equal')
    plt.tight_layout()
    
    return plt.gcf()

def analyze_stagnation_points(df, stagnation_points):
    """
    Print analysis of the stagnation points.
    """
    print(f"Found {len(stagnation_points)} stagnation points")
    print("\nStagnation points details:")
    print(stagnation_points[['x', 'y', 'u', 'v', 'p']].to_string())
    
    if len(stagnation_points) > 0:
        print("\nPressure statistics at stagnation points:")
        print(stagnation_points['p'].describe())

if __name__ == "__main__":
    # Read the data
    # Replace with your file path
    file_path = "NEW_DATA/data_new5.csv"
    df = pd.read_csv(file_path)
    
    # Find stagnation points
    stagnation_points = find_stagnation_points(df)
    
    # Analyze the stagnation points
    analyze_stagnation_points(df, stagnation_points)
    
    # Create the plot
    fig = plot_stagnation_points(df, stagnation_points)
    plt.show()
    
    # Save the plot if needed
    # fig.savefig('stagnation_points.png', dpi=300, bbox_inches='tight')
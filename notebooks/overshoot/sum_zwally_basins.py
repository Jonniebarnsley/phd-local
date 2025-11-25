import pandas as pd
import argparse
import sys
import os
from pathlib import Path

def sum_zwally_basins(input_directory, output_file):
    """
    Sum up all ZWALLY basin files to create a total Antarctica sea level rise file.
    
    Args:
        input_directory (str): Directory containing ZWALLY*.csv files
        output_file (str): Path for the output total file
    """
    
    # Find all ZWALLY*.csv files in the input directory
    input_path = Path(input_directory)
    zwally_files = list(input_path.glob("ZWALLY*.csv"))
    
    if not zwally_files:
        print(f"Error: No ZWALLY*.csv files found in {input_directory}")
        sys.exit(1)
    
    print(f"Found {len(zwally_files)} ZWALLY files:")
    for file in sorted(zwally_files):
        print(f"  - {file.name}")
    
    # Read the first file to get the structure
    first_df = pd.read_csv(zwally_files[0])
    print(f"\nFirst file shape: {first_df.shape}")
    print(f"Columns: {list(first_df.columns[:10])}...")  # Show first 10 columns
    
    # Initialize the sum DataFrame with the same structure as the first file
    total_df = first_df.copy()
    
    # Set all numeric columns to 0 (except the index column if it exists)
    numeric_columns = total_df.select_dtypes(include=['number']).columns
    total_df[numeric_columns] = 0
    
    # Sum up all files
    print("\nSumming up basin files...")
    for i, file_path in enumerate(sorted(zwally_files)):
        print(f"Processing {file_path.name} ({i+1}/{len(zwally_files)})")
        
        # Read the current file
        current_df = pd.read_csv(file_path)
        
        # Verify the structure matches
        if current_df.shape != first_df.shape:
            print(f"Warning: {file_path.name} has different shape {current_df.shape} vs {first_df.shape}")
            continue
        
        # Add numeric values to the total
        for col in numeric_columns:
            if col in current_df.columns:
                total_df[col] += current_df[col]
    
    # Save the total file
    total_df.to_csv(output_file, index=False)
    
    print(f"\nSummation complete!")
    print(f"Total shape: {total_df.shape}")
    print(f"Output saved to: {output_file}")
    
    # Show some statistics
    print(f"\nStatistics for year 2100 (column 2100):")
    year_2100_col = 2100
    if year_2100_col in total_df.columns:
        print(f"  Mean: {total_df[year_2100_col].mean():.6f}")
        print(f"  Min: {total_df[year_2100_col].min():.6f}")
        print(f"  Max: {total_df[year_2100_col].max():.6f}")
    
    return total_df

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Sum up all ZWALLY basin files to create total Antarctica sea level rise')
    parser.add_argument('input_directory', help='Directory containing ZWALLY*.csv files')
    parser.add_argument('output_file', help='Path for the output total file')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Check if input directory exists
        if not os.path.isdir(args.input_directory):
            print(f"Error: Input directory '{args.input_directory}' does not exist.")
            sys.exit(1)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Sum up the basin files
        total_data = sum_zwally_basins(args.input_directory, args.output_file)
        
        # Display first few rows for verification
        print("\nFirst 5 rows of total data:")
        print(total_data.head())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 
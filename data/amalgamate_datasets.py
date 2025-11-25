import pandas as pd
import argparse
import sys
import os
from pathlib import Path

def amalgamate_datasets(input_directories, output_directory):
    """
    Amalgamate datasets from multiple directories into a single directory.
    
    Args:
        input_directories (list): List of input directory paths
        output_directory (str): Path for the output directory
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_directory}")
    
    # Dictionary to store all data for each basin
    all_basin_data = {}
    
    # Process each input directory
    for input_dir in input_directories:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue
            
        print(f"\nProcessing directory: {input_dir}")
        
        # Find all ZWALLY*.csv files in the input directory
        # Handle both direct files and files in subdirectories like zwally_v5
        zwally_files = list(input_path.glob("ZWALLY*.csv"))
        zwally_v5_files = list(input_path.glob("zwally_v5/ZWALLY*.csv"))
        all_files = zwally_files + zwally_v5_files
        
        if not all_files:
            print(f"  No ZWALLY*.csv files found in {input_dir}")
            continue
            
        print(f"  Found {len(all_files)} ZWALLY files")
        
        # Process each ZWALLY file
        for file_path in sorted(all_files):
            basin_name = file_path.name  # e.g., "ZWALLY01.csv"
            
            print(f"    Processing {basin_name}...")
            
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Remove any "Unnamed: 0" index columns that might be present
                unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
                if unnamed_cols:
                    df = df.drop(columns=unnamed_cols)
                    print(f"      Removed {len(unnamed_cols)} unnamed index column(s)")
                
                # Convert from cm to m for more_overshoot and SSP245 directories
                # mira_by_basin data is already in m
                if 'more_overshoot' in str(input_dir) or 'SSP245' in str(input_dir):
                    # Find numeric columns (year columns) and convert from cm to m
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    # Exclude the key columns that shouldn't be converted
                    key_cols = ['gamma', 'scenario', 'gcm', 'slidinglaw']
                    year_cols = [col for col in numeric_cols if col not in key_cols]
                    
                    if year_cols:
                        df[year_cols] = df[year_cols] / 100.0  # Convert cm to m
                        print(f"      Converted {len(year_cols)} year columns from cm to m")
                
                # Initialize basin data if not exists
                if basin_name not in all_basin_data:
                    all_basin_data[basin_name] = []
                
                # Add data from this file to the basin collection
                all_basin_data[basin_name].append(df)
                
                print(f"      Added {len(df)} rows from {input_dir}")
                
            except Exception as e:
                print(f"      Error reading {file_path}: {e}")
                continue
    
    # Combine data for each basin and write to output directory
    print(f"\nCombining data and writing to {output_directory}...")
    
    for basin_name, dataframes in all_basin_data.items():
        if not dataframes:
            continue
            
        print(f"  Processing {basin_name}...")
        
        try:
            # Concatenate all dataframes for this basin
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Remove duplicates based on key columns (gamma, scenario, gcm, slidinglaw)
            # Keep the first occurrence of each unique combination
            key_columns = ['gamma', 'scenario', 'gcm', 'slidinglaw']
            
            # Check if all key columns exist
            missing_columns = [col for col in key_columns if col not in combined_df.columns]
            if missing_columns:
                print(f"    Warning: Missing columns {missing_columns} in {basin_name}, skipping deduplication")
            else:
                # Count duplicates before removal
                total_rows = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=key_columns, keep='first')
                removed_rows = total_rows - len(combined_df)
                
                if removed_rows > 0:
                    print(f"    Removed {removed_rows} duplicate rows")
            
            # Write the combined data to output file
            output_file = output_path / basin_name
            combined_df.to_csv(output_file, index=False)
            
            print(f"    Wrote {len(combined_df)} rows to {basin_name}")
            
        except Exception as e:
            print(f"    Error processing {basin_name}: {e}")
            continue
    
    print(f"\nAmalgamation complete! Output written to {output_directory}")

def main():
    parser = argparse.ArgumentParser(description='Amalgamate ZWALLY datasets from multiple directories')
    parser.add_argument('--input-dirs', nargs='+', required=True,
                       help='Input directories containing ZWALLY*.csv files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for amalgamated files')
    
    args = parser.parse_args()
    
    # Validate input directories
    for input_dir in args.input_dirs:
        if not os.path.exists(input_dir):
            print(f"Error: Input directory {input_dir} does not exist")
            sys.exit(1)
    
    # Run the amalgamation
    amalgamate_datasets(args.input_dirs, args.output_dir)

if __name__ == "__main__":
    main() 
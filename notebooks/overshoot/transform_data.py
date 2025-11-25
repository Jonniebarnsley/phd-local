import pandas as pd
import argparse
import sys

def transform_sea_level_data(input_file, output_file):
    """
    Transform sea level rise CSV from wide format to restructured format.
    
    Original format has columns like: overshoot_CESM2-WACCM_J300_MA
    New format will have separate columns for: scenario, model, sliding_law, basal_melt, and years 2007-2300
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get all column names except 'Year'
    data_columns = [col for col in df.columns if col != 'Year']
    
    # Create a list to store the transformed data
    transformed_rows = []
    
    # Parse each column name to extract components
    for col in data_columns:
        # Split the column name by underscore
        parts = col.split('_')
        
        if len(parts) >= 4:
            scenario = parts[0]  # e.g., 'overshoot'
            model = parts[1]     # e.g., 'CESM2-WACCM'
            sliding_law = parts[2]  # e.g., 'J300'
            basal_melt = parts[3]   # e.g., 'MA'
            
            # Convert scenario name
            if scenario == 'overshoot':
                scenario = 'ssp534-over'
            elif scenario == 'SSP245':
                scenario = 'ssp245'
            
            # Convert basal melt parameter names
            if basal_melt == 'MA':
                basal_melt = 'meanAnt'
            elif basal_melt == 'PIGL5th':
                basal_melt = 'PIG5'
            
            # Create a row for this parameter combination
            row_data = {
                'gamma': basal_melt,
                'scenario': scenario,
                'gcm': model,
                'slidinglaw': sliding_law
            }
            
            # Add data for each year
            for idx, year in enumerate(df['Year']):
                year_col = f'year_{int(year)}'
                row_data[year_col] = df[col].iloc[idx]
            
            transformed_rows.append(row_data)
    
    # Create the new DataFrame
    transformed_df = pd.DataFrame(transformed_rows)
    
    # Reorder columns to have metadata first, then years in order
    metadata_cols = ['gamma', 'scenario', 'gcm', 'slidinglaw']
    year_cols = sorted([col for col in transformed_df.columns if col.startswith('year_')])
    ordered_cols = metadata_cols + year_cols
    
    transformed_df = transformed_df[ordered_cols]
    
    # Save to new CSV
    transformed_df.to_csv(output_file, index=False)
    
    print("Transformation complete!")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {transformed_df.shape}")
    print(f"Output saved to: {output_file}")
    
    return transformed_df

# Alternative function if you want years as regular column names (2007, 2008, etc.)
def transform_sea_level_data_simple_years(input_file, output_file):
    """
    Same transformation but with year columns named as just the year (2007, 2008, etc.)
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get all column names except 'Year'
    data_columns = [col for col in df.columns if col != 'Year']
    
    # Create a list to store the transformed data
    transformed_rows = []
    
    # Parse each column name to extract components
    for col in data_columns:
        # Split the column name by underscore
        parts = col.split('_')
        
        if len(parts) >= 4:
            scenario = parts[0]
            model = parts[1]
            sliding_law = parts[2]
            basal_melt = parts[3]
            
            # Convert scenario name
            if scenario == 'overshoot':
                scenario = 'ssp534-over'
            elif scenario == 'SSP245':
                scenario = 'ssp245'
            
            # Convert basal melt parameter names
            if basal_melt == 'MA':
                basal_melt = 'meanAnt'
            elif basal_melt == 'PIGL5th':
                basal_melt = 'PIG5'
            
            # Create a row for this parameter combination
            row_data = {
                'gamma': basal_melt,
                'scenario': scenario,
                'gcm': model,
                'slidinglaw': sliding_law
            }
            
            # Add data for each year (using year as column name directly)
            for idx, year in enumerate(df['Year']):
                row_data[int(year)] = df[col].iloc[idx]
            
            transformed_rows.append(row_data)
    
    # Create the new DataFrame
    transformed_df = pd.DataFrame(transformed_rows)
    
    # Reorder columns to have metadata first, then years in order
    metadata_cols = ['gamma', 'scenario', 'gcm', 'slidinglaw']
    year_cols = sorted([col for col in transformed_df.columns if isinstance(col, int)])
    ordered_cols = metadata_cols + year_cols
    
    transformed_df = transformed_df[ordered_cols]
    
    # Save to new CSV
    transformed_df.to_csv(output_file, index=False)
    
    print(f"Transformation complete!")
    print(f"Original shape: {df.shape}")
    print(f"New shape: {transformed_df.shape}")
    print(f"Output saved to: {output_file}")
    
    return transformed_df

# Example usage
if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Transform sea level rise CSV from wide format to restructured format')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('output_file', help='Path to the output CSV file')
    parser.add_argument('--prefixed-years', action='store_true', 
                       help='Use prefixed year column names (year_2007, year_2008, etc.) instead of simple years (2007, 2008, etc.)')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Choose which transformation function to use based on arguments
        if args.prefixed_years:
            transformed_data = transform_sea_level_data(args.input_file, args.output_file)
        else:
            transformed_data = transform_sea_level_data_simple_years(args.input_file, args.output_file)
        
        # Display first few rows and columns for verification
        print("\nFirst 5 rows of transformed data:")
        print(transformed_data.head())
        
        print(f"\nTotal columns: {len(transformed_data.columns)}")
        print(f"Metadata columns: {transformed_data.columns[:4].tolist()}")
        print(f"Year range: {transformed_data.columns[4]} to {transformed_data.columns[-1]}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
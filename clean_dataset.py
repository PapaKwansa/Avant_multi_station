import pandas as pd
import numpy as np

def clean_comsol_strain_txt(input_file, output_file=None):
    """
    Clean COMSOL exported strain dataset for inversion.
    
    Parameters
    ----------
    input_file : str
        Path to the COMSOL txt file.
    output_file : str, optional
        If provided, saves the cleaned dataset to CSV.
        
    Returns
    -------
    df : pandas.DataFrame
        Cleaned dataset with time as first column, strain components as other columns.
    """
    
    # --- Step 1: Read the file, skipping comment lines ---
    # COMSOL uses % for headers, so skip those
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Keep only lines that start with a number (time column)
    data_lines = [line for line in lines if line.strip() and line[0].isdigit()]
    
    # --- Step 2: Split lines into columns ---
    data = []
    for line in data_lines:
        # Replace multiple spaces/tabs with single space
        row = [float(x) for x in line.strip().split()]
        data.append(row)
    
    data = np.array(data)
    
    # --- Step 3: Build column names ---
    # First column is time
    n_strains = data.shape[1] - 1
    strain_cols = [f"strain_{i+1}" for i in range(n_strains)]
    columns = ["time_s"] + strain_cols
    
    # --- Step 4: Create DataFrame ---
    df = pd.DataFrame(data, columns=columns)
    
    # --- Step 5: Optional: save to CSV ---
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"[INFO] Cleaned dataset saved to: {output_file}")
    
    return df

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    input_file = "Avant_array.txt"
    output_file = "Avant_cleaned_strain.csv"
    
    df_clean = clean_comsol_strain_txt(input_file, output_file)
    
    print("[INFO] Dataset shape:", df_clean.shape)
    print(df_clean.head())
import argparse
import pandas as pd
import plotly.graph_objects as go
import os
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing scaling CSVs")
    parser.add_argument("--out_html", type=str, required=True, help="Output HTML filepath")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_dir):
        print(f"Error: Directory {args.csv_dir} does not exist.")
        exit(1)
        
    csv_files = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    if not csv_files:
        print(f"No CSVs found in {args.csv_dir}")
        exit(1)
        
    fig = go.Figure()
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        name = filename.replace('.csv', '').upper()
        
        df = pd.read_csv(csv_file)
        # Drop rows where WallTime is ERROR
        df = df[df['WallTime'] != 'ERROR'].copy()
        
        if df.empty or 'N' not in df.columns or 'WallTime' not in df.columns:
            print(f"Skipping {filename} due to missing or invalid columns")
            continue
            
        df['WallTime'] = pd.to_numeric(df['WallTime'])
            
        fig.add_trace(go.Scatter(
            x=df['N'],
            y=df['WallTime'],
            mode='lines+markers',
            name=name
        ))
        
    fig.update_layout(
        title="GMM Execution Scaling (Wall Time vs Dataset Size)",
        xaxis_title="Dataset Size (N)",
        yaxis_title="Total Wall Time (Seconds)",
        hovermode="x unified"
    )
    
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    fig.write_html(args.out_html)
    print(f"Scaling chart saved to {args.out_html}")

if __name__ == "__main__":
    main()

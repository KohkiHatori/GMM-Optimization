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
        
    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        print(f"No CSVs found in {args.csv_dir}")
        exit(1)
        
    fig = go.Figure()
    
    # Define a list of symbols for different lines
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'pentagon', 'hexagon', 'star']
    
    for i, csv_file in enumerate(csv_files):
        filename = os.path.basename(csv_file)
        name = filename.replace('.csv', '').upper()
        
        df = pd.read_csv(csv_file)
        # Drop rows where WallTime is ERROR
        df = df[df['WallTime'] != 'ERROR'].copy()
        
        # Ensure N and WallTime are numeric and drop invalid rows
        df['N'] = pd.to_numeric(df['N'], errors='coerce')
        df['WallTime'] = pd.to_numeric(df['WallTime'], errors='coerce')
        df = df.dropna(subset=['N', 'WallTime'])
        
        if df.empty:
            print(f"Skipping {filename} due to no valid numeric data")
            continue
            
        # Sort by N to ensure lines are drawn correctly
        df = df.sort_values('N')
            
        fig.add_trace(go.Scatter(
            x=df['N'],
            y=df['WallTime'],
            mode='lines+markers',
            name=name,
            marker=dict(symbol=symbols[i % len(symbols)], size=10),
            line=dict(width=3)
        ))
        
    fig.update_layout(
        title=dict(
            text="GMM Execution Scaling (Wall Time vs Dataset Size)",
            font=dict(size=24)
        ),
        xaxis=dict(
            type='log', 
            gridcolor='lightgray', 
            gridwidth=1, 
            tickfont=dict(size=14), 
            dtick=1, # Show only powers of 10 (1, 10, 100...)
            title=dict(text="Dataset Size (N)", font=dict(size=18))
        ),
        yaxis=dict(
            type='log', 
            gridcolor='lightgray', 
            gridwidth=1, 
            tickfont=dict(size=14), 
            dtick=1, # Show only powers of 10 (0.1, 1, 10...)
            tickformat="~g", # Use general format to show clean numbers
            title=dict(text="Total Wall Time (Seconds)", font=dict(size=18))
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16), borderwidth=1)
    )
    
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    fig.write_html(args.out_html)
    print(f"Scaling chart saved to {args.out_html}")

if __name__ == "__main__":
    main()

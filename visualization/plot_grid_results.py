import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing grid scaling CSVs")
    parser.add_argument("--out_html", type=str, required=True, help="Output HTML filepath")
    args = parser.parse_args()
    
    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        print(f"No CSVs found in {args.csv_dir}")
        exit(1)
        
    fig = make_subplots(
        rows=1, cols=len(csv_files),
        subplot_titles=[os.path.basename(f).replace('.csv', '').upper() for f in csv_files],
        shared_yaxes=True,
        horizontal_spacing=0.05
    )
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        df = df[df['WallTime'] != 'ERROR'].copy()
        df['WallTime'] = pd.to_numeric(df['WallTime'])
        
        # Create pivot for heatmap: Rows=D, Cols=K
        pivot = df.pivot(index='D', columns='K', values='WallTime')
        
        heatmap = go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            coloraxis="coloraxis",
            hovertemplate="D: %{y}<br>K: %{x}<br>Time: %{z:.4f}s<extra></extra>"
        )
        
        fig.add_trace(heatmap, row=1, col=i+1)
        
    fig.update_layout(
        title="GMM Performance Landscape (D vs K)",
        coloraxis=dict(colorscale='Viridis', colorbar_title="Seconds"),
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Dimensions (D)",
        template="plotly_white",
        height=500
    )
    
    # Update all x-axes
    for j in range(len(csv_files)):
        fig.update_xaxes(title_text="K", row=1, col=j+1)
    
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.write_html(args.out_html)
    print(f"Grid analysis saved to {args.out_html}")

if __name__ == "__main__":
    main()
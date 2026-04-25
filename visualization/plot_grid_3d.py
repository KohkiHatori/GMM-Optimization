import argparse
import pandas as pd
import plotly.graph_objects as go
import os
import glob
from scipy.interpolate import griddata
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing grid scaling CSVs")
    parser.add_argument("--out_html", type=str, required=True, help="Output HTML filepath")
    args = parser.parse_args()
    
    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if not csv_files:
        print(f"No CSVs found in {args.csv_dir}")
        exit(1)
        
    fig = go.Figure()
    
    # Define colorscales for different implementations to easily distinguish them
    colorscales = ['Blues', 'Reds', 'Greens', 'Purples', 'Oranges']
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        df = df[df['WallTime'] != 'ERROR'].copy()
        df['WallTime'] = pd.to_numeric(df['WallTime'])
        
        name = os.path.basename(csv_file).replace('.csv', '').upper()
        
        # We use griddata to interpolate the points into a smooth surface, 
        # which looks much better than a blocky wireframe.
        x = df['K'].values
        y = df['D'].values
        z = df['WallTime'].values
        
        # Create a dense grid for smooth interpolation
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        X, Y = np.meshgrid(xi, yi)
        
        Z = griddata((x, y), z, (X, Y), method='cubic')
        
        # Add the 3D surface
        fig.add_trace(go.Surface(
            x=X, 
            y=Y, 
            z=Z, 
            name=name,
            colorscale=colorscales[i % len(colorscales)],
            opacity=0.8,
            showscale=False, # Hide individual colorbars to keep it clean
            hovertemplate=f"<b>{name}</b><br>K: %{{x:.0f}}<br>D: %{{y:.0f}}<br>Time: %{{z:.4f}}s<extra></extra>"
        ))
        
        # Add the actual data points as scatter points floating on the surface
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            name=f"{name} (Data)",
            marker=dict(size=4, color='black', symbol='circle', opacity=0.5),
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(text="GMM Performance Scaling: 3D Surface Analysis", font=dict(size=24)),
        scene=dict(
            xaxis_title=dict(text="Clusters (K)", font=dict(size=14)),
            yaxis_title=dict(text="Dimensions (D)", font=dict(size=14)),
            zaxis_title=dict(text="Wall Time (Seconds)", font=dict(size=14)),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=1.0) # Start with a good isometric view
            ),
            xaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            yaxis=dict(gridcolor='lightgray', backgroundcolor='white'),
            zaxis=dict(gridcolor='lightgray', backgroundcolor='white')
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            font=dict(size=16),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, b=0, t=50) # Tighter margins for 3D plots
    )
    
    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.write_html(args.out_html)
    print(f"3D Surface plot saved to {args.out_html}")

if __name__ == "__main__":
    main()
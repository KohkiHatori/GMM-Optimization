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

    all_data = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df['WallTime'] != 'ERROR'].copy()
        df['WallTime'] = pd.to_numeric(df['WallTime'])
        name = os.path.basename(csv_file).replace('.csv', '').upper()
        df['Implementation'] = name
        all_data.append(df)

    df_full = pd.concat(all_data, ignore_index=True)
    df_full = df_full[df_full['K'] <= 16].copy()
    df_full = df_full[df_full['D'] == 32].copy()
    sorted_d = sorted(df_full['D'].unique())
    implementations = sorted(df_full['Implementation'].unique())

    # Use a consistent color map and marker symbols for implementations across all subplots
    colors = px.colors.qualitative.Set1 # Using a different set for distinct implementations
    color_map = {impl: colors[i % len(colors)] for i, impl in enumerate(implementations)}

    # Symbols for color-blind accessibility
    symbols = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'pentagon']
    symbol_map = {impl: symbols[i % len(symbols)] for i, impl in enumerate(implementations)}

    fig = make_subplots(
        rows=len(sorted_d), cols=1,
        subplot_titles=[f"Dimensions D = {d}" for d in sorted_d],
        vertical_spacing=0.05
    )

    for i, d_val in enumerate(sorted_d):
        d_df = df_full[df_full['D'] == d_val]

        for impl in implementations:
            impl_df = d_df[d_df['Implementation'] == impl].sort_values('K')
            if impl_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=impl_df['K'],
                    y=impl_df['WallTime'],
                    name=impl,
                    legendgroup=impl,
                    showlegend=(i == 0), # Only show legend once
                    line=dict(color=color_map[impl], width=4),
                    mode='lines+markers',
                    marker=dict(
                        size=20,
                        symbol=symbol_map[impl],
                        line=dict(width=2, color='white')
                    ),
                    hovertemplate=f"D: {d_val}<br>Impl: {impl}<br>K: %{{x}}<br>Time: %{{y:.4f}}s<extra></extra>"
                ),
                row=i+1, col=1
            )

        fig.update_xaxes(title_text="Number of Clusters (K)", row=i+1, col=1)
        fig.update_yaxes(title_text="Wall Time (seconds)", row=i+1, col=1)

    fig.update_layout(
        title=dict(
            text="GMM Performance Scaling: Wall Time vs K (D=32)",
            font=dict(size=24, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        template="plotly_white",
        height=800,
        width=1200,
        legend_title="Implementation",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=dict(size=24),
            title=dict(font=dict(size=26))
        ),
        margin=dict(t=100, b=100, l=100, r=100),
        font=dict(family="Inter, Roboto, Helvetica Neue, sans-serif")
    )

    out_dir = os.path.dirname(args.out_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.write_html(args.out_html)
    print(f"Line scaling analysis saved to {args.out_html}")

if __name__ == "__main__":
    main()
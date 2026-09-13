"""Data-aligned Plotly figures for the AI-RAN allocation demo."""
import numpy as np
import plotly.graph_objects as go


def rate_surface(data, selected, T_ms, scenario_id):
    """z always shows current FBL sum-rate; sparse rays remain sparse."""
    fig = go.Figure()
    if "allocation_ratio" not in data.columns:
        grid = data.pivot(index="t2", columns="t1", values="current").sort_index().sort_index(axis=1)
        fig.add_trace(go.Surface(
            x=grid.columns.to_numpy()*T_ms/100, y=grid.index.to_numpy()*T_ms/100, z=grid.to_numpy(),
            colorscale=[[0, "#162643"], [.45, "#297F94"], [1, "#55E2BF"]],
            colorbar=dict(title="bit/s/Hz", thickness=12), connectgaps=False,
            contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
            hovertemplate="Precoder: %{x:g} ms<br>Quantizer: %{y:g} ms<br>FBL sum-rate: %{z:.6f} bit/s/Hz<extra></extra>",
            name="Evaluated grid",
        ))
        for axis, other, color in [("t1", "t2", "#36D6BF"), ("t2", "t1", "#A99BFF")]:
            rows = data[np.isclose(data[other], selected[other])].sort_values(axis)
            fig.add_trace(go.Scatter3d(x=rows.t1*T_ms/100, y=rows.t2*T_ms/100, z=rows.current,
                mode="lines", line=dict(color=color, width=6), name=f"Selected {other} slice"))
    else:
        for ratio, rows in data.groupby("allocation_ratio"):
            rows = rows.sort_values("total")
            fig.add_trace(go.Scatter3d(x=rows.t1*T_ms/100, y=rows.t2*T_ms/100, z=rows.current,
                mode="lines+markers", name=f"t₁:t₂ = {ratio}", marker=dict(size=4)))
    best = data.loc[data.current.idxmax()]
    for row, name, color, symbol in [(selected, "Current allocation", "#F4C66B", "circle"),
                                      (best, "Highest FBL sum-rate", "#FFFFFF", "diamond")]:
        fig.add_trace(go.Scatter3d(x=[row.t1*T_ms/100], y=[row.t2*T_ms/100], z=[row.current],
            mode="markers", name=name, marker=dict(color=color, size=7, symbol=symbol),
            hovertemplate="Precoder: %{x:g} ms<br>Quantizer: %{y:g} ms<br>FBL sum-rate: %{z:.6f} bit/s/Hz<extra>%{fullData.name}</extra>"))
    fig.update_layout(template="plotly_dark", height=520, paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0), font=dict(color="#B9C8DC"),
        legend=dict(orientation="h", y=1.03),
        scene=dict(xaxis_title="Precoder inference time, t₁ (ms)",
                   yaxis_title="Quantizer inference time, t₂ (ms)",
                   zaxis_title="Finite-blocklength<br>sum-rate (bit/s/Hz)",
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.1)), aspectmode="cube",
                   uirevision=scenario_id),
    )
    return fig

"""Explicit resource estimates for saved inference budgets (not telemetry)."""
import numpy as np
import plotly.graph_objects as go

MIB = 1024 ** 2


def tensor_storage(data, resources, retain_all=True):
    """One independent candidate bank per stage, complex64, batch=1.

    Quantizer storage counts one input sample bank plus retained output banks.
    This deliberately excludes activation and Bussgang/WMMSE workspace memory.
    """
    result = data.copy()
    c1 = result["precoder_candidates"].astype(float)
    c2 = result["quantizer_candidates"].astype(float)
    if not retain_all:
        c1, c2 = c1.clip(upper=1), c2.clip(upper=1)
    element = resources["complex_element_bytes"]
    result["precoder_buffers_mib"] = c1 * resources["precoder_elements"] * element / MIB
    result["quantizer_buffers_mib"] = (
        (1 + c2) * result["mc_signal_samples"] * resources["signal_elements"] * element / MIB
    )
    result["weights_mib"] = (resources["precoder_weights_bytes"] + resources["quantizer_weights_bytes"]) / MIB
    result["buffers_mib"] = result.precoder_buffers_mib + result.quantizer_buffers_mib
    result["accounted_mib"] = result.weights_mib + result.buffers_mib
    return result


def memory_curve(scenario, estimated, selected):
    """Link memory to saved allocation points without inventing extra samples."""
    fig = go.Figure()
    if "allocation_ratio" not in estimated.columns:
        sweeps = [
            (estimated[np.isclose(estimated.t2, selected.t2)], "Vary precoder time", "#36D6BF"),
            (estimated[np.isclose(estimated.t1, selected.t1)], "Vary quantizer time", "#A99BFF"),
        ]
    else:
        sweeps = [(estimated[estimated["allocation_ratio"] == selected["allocation_ratio"]],
                   "Current allocation ratio", "#36D6BF")]
    for rows, name, color in sweeps:
        rows = rows.sort_values("total")
        fig.add_trace(go.Scatter(
            x=rows.total * scenario["T_ms"] / 100, y=rows.accounted_mib,
            mode="lines+markers", name=name, line=dict(color=color, width=2),
            hovertemplate="Inference time: %{x:g} ms<br>Estimated memory: %{y:.3f} MiB<extra>%{fullData.name}</extra>",
        ))
    row = estimated.loc[selected.name]
    fig.add_trace(go.Scatter(
        x=[selected.total * scenario["T_ms"] / 100], y=[row.accounted_mib],
        mode="markers", name="Selected", marker=dict(color="#F4C66B", size=12),
        hovertemplate="%{x:g} ms · %{y:.3f} MiB<extra>Selected</extra>",
    ))
    return fig

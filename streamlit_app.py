"""AI-RAN experiment explorer. Run: streamlit run streamlit_app.py"""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard_model import DATA_PATH, METRICS, best_point, frame, load_experiments, select_point
from dashboard_resources import memory_curve, tensor_storage
from dashboard_plots import rate_surface

st.set_page_config(page_title="AI-RAN Inference Time Allocation", page_icon="📡", layout="wide")

TEAL, PURPLE, GOLD = "#36D6BF", "#A99BFF", "#F4C66B"
COLORS = [TEAL, PURPLE, GOLD]
st.markdown("""<style>
.block-container {max-width:1500px; padding-top:2rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {border-right:1px solid #213049;}
[data-testid="stMetric"] {background:#111C2D; border:1px solid #233149;
border-radius:14px; padding:16px 19px; min-height:120px;}
[data-testid="stMetricLabel"] {color:#9DAEC6;}
.eyebrow {color:#36D6BF; font-size:11px; letter-spacing:3px; font-weight:700;}
.hero-title {font-size:38px; font-weight:750; letter-spacing:-1.5px; margin:5px 0;}
.subtitle {color:#9DAEC6; font-size:15px; margin-bottom:22px;}
.tag {display:inline-block; border:1px solid #2B4353; border-radius:20px;
padding:5px 12px; color:#BDEFE6; font-size:12px; margin:0 6px 8px 0;}
.section-label {font-size:11px; letter-spacing:2px; color:#8C9FB9; margin-top:20px;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def read_data(modified_ns):
    return load_experiments()


def style(fig, height=330, x_title=None, y_title=None):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0D1727", margin=dict(l=12, r=15, t=35, b=20),
        font=dict(family="Arial, sans-serif", color="#B9C8DC", size=12),
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=11)),
        xaxis_title=x_title, yaxis_title=y_title,
    )
    fig.update_xaxes(gridcolor="#203047", zeroline=False)
    fig.update_yaxes(gridcolor="#203047", zeroline=False)
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", key=key, config={
        "displaylogo": False, "toImageButtonOptions": {"format": "svg", "filename": key},
    })


def move_to(point, scenario_id, experiment):
    if experiment != "scalability":
        st.session_state[f"{scenario_id}_t1"] = float(point["t1"])
        st.session_state[f"{scenario_id}_t2"] = float(point["t2"])
    else:
        st.session_state[f"{scenario_id}_ratio"] = point["allocation_ratio"]
        st.session_state[f"{scenario_id}_total"] = float(point["total"])


def timeline(point, T):
    fig = go.Figure()
    for name, value, color in [
        ("Precoder t₁", point.t1, TEAL), ("Quantizer t₂", point.t2, PURPLE),
        ("Unallocated", max(0, 100 - point.total), "#34465E"),
    ]:
        fig.add_trace(go.Bar(x=[value * T / 100], y=["Budget"], orientation="h",
                             name=name, marker_color=color,
                             text=[f"{name}  {value*T/100:g} ms"], textposition="inside",
                             hovertemplate="%{x:g} ms<extra>%{fullData.name}</extra>"))
    style(fig, 130, "ms")
    fig.update_layout(barmode="stack", margin=dict(l=0, r=0, t=0, b=20), showlegend=False)
    fig.update_xaxes(range=[0, T])
    fig.update_yaxes(visible=False)
    return fig


def topology(scenario, point):
    fig = go.Figure()
    n_ru, k = scenario["num_rus"], scenario["num_users"]
    ru_y, ue_y = np.linspace(.15, .85, n_ru), np.linspace(.05, .95, k)
    for y in ru_y:
        fig.add_trace(go.Scatter(x=[.13, .5], y=[.5, y], mode="lines",
                                 line=dict(color="#3C6478", width=2), hoverinfo="skip"))
        for u in ue_y:
            fig.add_trace(go.Scatter(x=[.5, .86], y=[y, u], mode="lines",
                                     line=dict(color="rgba(100,132,180,.19)", width=1), hoverinfo="skip"))
    for x, ys, labels, color, symbol, size in [
        (.13, [.5], ["DU"], PURPLE, "square", 38),
        (.5, ru_y, [f"RU {i+1}" for i in range(n_ru)], TEAL, "diamond", 22),
        (.86, ue_y, [f"UE {i+1}" for i in range(k)], GOLD, "circle", 12),
    ]:
        fig.add_trace(go.Scatter(x=[x]*len(ys), y=list(ys), mode="markers+text",
                                 text=labels, textposition="middle right",
                                 marker=dict(color=color, size=size, symbol=symbol),
                                 hovertemplate="%{text}<extra>Topology schematic</extra>"))
    fig.add_annotation(x=.13, y=.1, text=f"t₁ {point.t1:g}% · t₂ {point.t2:g}%", showarrow=False)
    style(fig, 300)
    fig.update_layout(showlegend=False, margin=dict(l=5, r=20, t=10, b=10))
    fig.update_xaxes(visible=False, range=[0, 1.06])
    fig.update_yaxes(visible=False, range=[-.02, 1.02])
    return fig


def allocation_map(data, selected, best, metric, T, experiment):
    fig = go.Figure()
    if experiment != "scalability":
        grid = data.pivot(index="t2", columns="t1", values=metric).sort_index().sort_index(axis=1)
        fig.add_trace(go.Heatmap(
            x=grid.columns.to_numpy()*T/100, y=grid.index.to_numpy()*T/100,
            z=grid.to_numpy(), colorscale=[[0, "#15233F"], [.5, "#286780"], [1, TEAL]],
            colorbar=dict(title="bit/s/Hz", thickness=10), hoverongaps=False,
            hovertemplate="t₁=%{x:g} ms<br>t₂=%{y:g} ms<br>Rate=%{z:.5f}<extra>Saved simulation</extra>",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data.t1*T/100, y=data.t2*T/100, mode="markers",
            marker=dict(size=13, color=data[metric], colorscale="Viridis", showscale=True,
                        colorbar=dict(title="bit/s/Hz", thickness=10)),
            customdata=data[metric], name="Saved points",
            hovertemplate="t₁=%{x:g} ms<br>t₂=%{y:g} ms<br>Rate=%{customdata:.5f}<extra></extra>",
        ))
    for row, symbol, color, name in [(best, "star", GOLD, "Grid best"),
                                    (selected, "circle-open", "white", "Selected")]:
        fig.add_trace(go.Scatter(x=[row.t1*T/100], y=[row.t2*T/100], mode="markers",
                                 marker=dict(symbol=symbol, size=19, color=color, line=dict(width=2)), name=name))
    return style(fig, 345, "Precoder t₁ (ms)", "Quantizer t₂ (ms)")


def rate_slice(data, selected, metric, T, axis, envelope=False):
    other = "t2" if axis == "t1" else "t1"
    values = data[np.isclose(data[other], selected[other])].sort_values(axis)
    fig = go.Figure(go.Scatter(x=values[axis]*T/100, y=values[metric], mode="lines+markers",
                               line=dict(color=TEAL if axis == "t1" else PURPLE, width=3),
                               name="Saved points", hovertemplate="%{x:g} ms<br>%{y:.5f} bit/s/Hz<extra></extra>"))
    fig.add_trace(go.Scatter(x=[selected[axis]*T/100], y=[selected[metric]], mode="markers",
                             marker=dict(color=GOLD, size=12), name="Selected"))
    if envelope:
        fig.add_trace(go.Scatter(x=values[axis]*T/100, y=values[metric].cummax(),
                                 mode="lines", line=dict(color="white", dash="dot", shape="hv"),
                                 name="Best saved rate ≤ budget"))
    return style(fig, 290, f"{'Precoder t₁' if axis == 't1' else 'Quantizer t₂'} (ms)", "bit/s/Hz")


try:
    scenarios = read_data(DATA_PATH.stat().st_mtime_ns)
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Could not load experiment data: {exc}")
    st.info("Run `python export_dashboard_data.py` in the research environment to export the experiment data.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="eyebrow">AI-RAN LAB</div>', unsafe_allow_html=True)
    st.title("Simulation settings")
    modes = [m for m in ("fixed_bank", "joint", "scalability") if any(s["experiment"] == m for s in scenarios)]
    if "fixed_bank" in modes and not st.session_state.get("fixed_grid_default_applied"):
        st.session_state["demo_evaluation_v2"] = "fixed_bank"
        st.session_state["fixed_grid_default_applied"] = True
    experiment = st.selectbox("Evaluation", modes, key="demo_evaluation_v2",
                              format_func=lambda x: {"fixed_bank": "Fixed-candidate inference", "joint": "Legacy allocation results", "scalability": "Joint budget sensitivity"}[x])
    available = [s for s in scenarios if s["experiment"] == experiment]
    versions = sorted({s["cache_version"] for s in available}, reverse=True)
    version = st.selectbox("Experiment version", versions, key=f"{experiment}_version")
    available = [s for s in available if s["cache_version"] == version]
    users = sorted({s["num_users"] for s in available})
    k = st.selectbox("Number of users K", users, index=users.index(4) if 4 in users else 0, key=f"{experiment}_{version}_k")
    available = [s for s in available if s["num_users"] == k]
    rhos = sorted({s["rho"] for s in available})
    rho = st.selectbox("Channel correlation ρ", rhos, index=rhos.index(.95) if .95 in rhos else len(rhos)-1,
                       key=f"{experiment}_{version}_{k}_rho")
    available = [s for s in available if s["rho"] == rho]
    scenario = st.selectbox("Saved run", available, format_func=lambda s: f"T={s['T_ms']:g} ms · {s['id'][:8]}") if len(available) > 1 else available[0]
    sid, T = scenario["id"], scenario["T_ms"]
    st.number_input("Coherence time T (ms)", value=float(T), disabled=True)
    if experiment == "joint":
        st.caption(f"Time conversion uses T = {T:g} ms from the training checkpoint. Legacy pair caches do not record runtime T.")
    else:
        st.caption("T is fixed by the saved experiment. Other coherence times require additional experiments.")
    metric = st.radio("Rate metric", scenario["metrics"], format_func=lambda m: METRICS[m], key=f"{experiment}_metric")
    st.divider()
    st.caption("Select the precoder and quantizer time budgets to update all rate and memory plots.")

data = frame(scenario)
best = best_point(data, metric)
st.markdown('<div class="eyebrow">CELL-FREE MASSIVE MIMO · AI-RAN DEMO</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">AI-RAN Inference Time Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Joint precoder and quantizer inference: finite-blocklength sum-rate and memory usage.</div>', unsafe_allow_html=True)
st.markdown(f'<span class="tag">K = {k} users</span><span class="tag">{scenario["num_rus"]} RUs</span>'
            f'<span class="tag">T = {T:g} ms</span><span class="tag">ρ = {rho:g}</span>'
            f'<span class="tag">{len(data)} saved points</span>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown("**Precoder and quantizer time budgets**")
    controls = st.columns([2, 2, 1])
    if experiment != "scalability":
        st.session_state.setdefault(f"{sid}_t1", 20.0 if 20 in data.t1.values else float(data.t1.min()))
        st.session_state.setdefault(f"{sid}_t2", 20.0 if 20 in data.t2.values else float(data.t2.min()))
        with controls[0]:
            t1 = st.select_slider("Precoder t₁", options=sorted(data.t1.unique().tolist()),
                                  format_func=lambda v: f"{v*T/100:g} ms ({v:g}% T)", key=f"{sid}_t1")
        with controls[1]:
            t2 = st.select_slider("Quantizer t₂", options=sorted(data.t2.unique().tolist()),
                                  format_func=lambda v: f"{v*T/100:g} ms ({v:g}% T)", key=f"{sid}_t2")
    else:
        st.session_state.setdefault(f"{sid}_ratio", "0.5:0.5")
        st.session_state.setdefault(f"{sid}_total", 40.0)
        with controls[0]:
            ratio = st.select_slider("Allocation ratio t₁ : t₂", options=sorted(data.allocation_ratio.unique()),
                                     key=f"{sid}_ratio")
        with controls[1]:
            total = st.select_slider("Total inference budget (t₁+t₂)/T", options=sorted(data.total.unique().tolist()),
                                     format_func=lambda v: f"{v:g}% · {v*T/100:g} ms", key=f"{sid}_total")
        row = data[(data.allocation_ratio == ratio) & np.isclose(data.total, total)].iloc[0]
        t1, t2 = row.t1, row.t2
    with controls[2]:
        st.button("Apply best on full grid", on_click=move_to, args=(best.to_dict(), sid, experiment), width="stretch")
    try:
        selected = select_point(data, t1, t2)
    except ValueError as exc:
        st.warning(str(exc))
        st.stop()
    show(timeline(selected, T), "allocation_timeline")
    st.caption("Unallocated time: $T-t_1-t_2$. This is a budget breakdown, not a transmission schedule or effective-rate formula.")

budget_best = best_point(data, metric, selected.total)
gap = float(budget_best[metric] - selected[metric])
resources = scenario.get("resources")
memory_data = None
if (resources and {"precoder_candidates", "quantizer_candidates", "mc_signal_samples"}.issubset(data.columns)
        and data[["precoder_candidates", "quantizer_candidates", "mc_signal_samples"]].notna().all().all()):
    memory_data = tensor_storage(data, resources)
metrics = st.columns(4)
metrics[0].metric("Selected sum-rate · bit/s/Hz", f"{selected[metric]:.5f}")
metrics[1].metric("Allocated inference time", f"{selected.total*T/100:g} ms", f"{selected.total:g}% of T", delta_color="off")
metrics[2].metric("Best under budget · bit/s/Hz", f"{budget_best[metric]:.5f}", f"Available gain +{gap:.5f}", delta_color="off")
metrics[3].metric("Estimated memory", f"{memory_data.loc[selected.name].accounted_mib:.3f} MiB" if memory_data is not None else "Not recorded",
                  help="Model tensors plus retained candidate buffers, not measured peak RAM/VRAM. See Data & provenance for assumptions.")
st.caption(f"Evaluated candidates: precoder {selected.precoder_candidates:g} · quantizer {selected.quantizer_candidates:g}.")
st.caption(METRICS[metric] + " · All rates are sum-rates in bit/s/Hz.")
if metric == "effective":
    st.info("Effective rate uses the original old/new-solution segment model and budget-dependent weights. These are raw pair-cache results, not simply (1−t₁/T−t₂/T) × rate.")
elif metric == "modeled":
    st.info("Sensitivity model: an assumed budget penalty is applied to the saved raw search rate. This is not measured effective throughput.")
if experiment == "fixed_bank":
    st.caption(f"Fixed-candidate evaluation · {scenario['evaluation_samples']} held-out samples × "
               f"{scenario['evaluation_blocks_per_sample']} channel blocks. More budget retains earlier candidates; rates can improve or plateau.")

overview, source_tab = st.tabs(["Simulation results", "Experiment details"])
with overview:
    left, right = st.columns([1, 1.35])
    with left, st.container(border=True):
        st.markdown("**Cell-free downlink configuration**")
        show(topology(scenario, selected), "network_topology")
        st.caption("Cell-free connectivity schematic. Locations, beams and per-user SINR are not measured.")
    with right, st.container(border=True):
        st.markdown("**Sum-rate over time allocations**")
        show(allocation_map(data, selected, best, metric, T, experiment), "allocation_landscape")
        st.caption("Circle: selected. Star: best on the full saved grid. " +
                   ("Each cell is one evaluated point." if experiment != "scalability" else "Only evaluated points on the three allocation ratios are shown."))
    if experiment != "scalability":
        if experiment == "joint":
            st.warning("Legacy results use changing candidate and evaluation conditions and the old covariance estimator. "
                       "Use Fixed-candidate inference to study the effect of additional search time.")
        envelope = st.checkbox("Show best saved rate within each budget", value=False) if experiment == "joint" else False
        if envelope:
            st.caption("Dotted line: maximum saved rate among allocations using no more than the x-axis budget, "
                       "with the other stage fixed. It assumes a smaller allocation may be reused; it is not a rerun at a larger allocation.")
        left, right = st.columns(2)
        with left, st.container(border=True):
            st.markdown(f"**Precoder inference time vs. sum-rate** · t₂ = {selected.t2*T/100:g} ms")
            show(rate_slice(data, selected, metric, T, "t1", envelope), "precoder_sweep")
        with right, st.container(border=True):
            st.markdown(f"**Quantizer inference time vs. sum-rate** · t₁ = {selected.t1*T/100:g} ms")
            show(rate_slice(data, selected, metric, T, "t2", envelope), "quantizer_sweep")
    else:
        with st.container(border=True):
            st.markdown("**Total inference time vs. sum-rate**")
            fig = go.Figure()
            for i, (ratio, rows) in enumerate(data.groupby("allocation_ratio")):
                rows = rows.sort_values("total")
                fig.add_trace(go.Scatter(x=rows.total, y=rows[metric], mode="lines+markers",
                                         name=f"t₁:t₂ = {ratio}", line=dict(color=COLORS[i], width=3)))
            fig.add_trace(go.Scatter(x=[selected.total], y=[selected[metric]], mode="markers",
                                     marker=dict(size=15, color="white", symbol="circle-open"), name="Selected"))
            show(style(fig, 320, "Joint inference budget (% T)", "bit/s/Hz"), "scalability_curves")
    with st.container(border=True):
        st.markdown("**3D finite-blocklength sum-rate**")
        show(rate_surface(data, selected, T, sid), "fbl_rate_surface")
        st.caption("Drag to rotate. The highlighted point follows the time sliders. " +
                   ("The surface connects evaluated grid points; intermediate values are not additional measurements."
                    if experiment != "scalability" else "Only evaluated allocation paths are shown; the gaps are not filled."))
    with st.container(border=True):
        st.markdown("**Sum-rate by allocation strategy**")
        comparison = [selected, budget_best, best]
        fig = go.Figure(go.Bar(
            x=[r[metric] for r in comparison],
            y=["My allocation", f"Best ≤ {selected.total*T/100:g} ms", "Best on full saved grid"],
            orientation="h", marker_color=[TEAL, PURPLE, GOLD],
            text=[f"{r[metric]:.5f} · ({r.t1:g}%, {r.t2:g}%)" for r in comparison], textposition="auto",
        ))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 225, "bit/s/Hz"), "allocation_comparison")
        st.button("Apply best under current budget", on_click=move_to,
                  args=(budget_best.to_dict(), sid, experiment))
        st.caption("Optima refer to the selected metric and saved grid, not the continuous allocation domain.")
        st.caption(f"Budget-limited best: t₁+t₂ ≤ {selected.total*T/100:g} ms "
                   f"(selected best uses {budget_best.total*T/100:g} ms). Full-grid best: searches every saved allocation "
                   f"and uses {best.total*T/100:g} ms, which may exceed your current budget.")
    if memory_data is not None:
        with st.container(border=True):
            st.markdown("**Inference time vs. memory**")
            show(style(memory_curve(scenario, memory_data, selected), 220,
                       "Total inference time (ms)", "Estimated memory (MiB)"), "memory_vs_inference_time")
            st.caption("Estimated model + candidate memory. The highlighted point follows the time sliders.")

with source_tab:
    st.markdown("**Evaluation data and assumptions**")
    st.write(scenario["provenance"])
    st.json({key: value for key, value in scenario.items() if key != "points"})
    st.caption(f"Selected source: results/{selected.source}")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button("Download scenario CSV", data.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"airan_{sid}.csv", mime="text/csv")
    snapshot = {"scenario": {k: v for k, v in scenario.items() if k != "points"},
                "metric": metric, "selected": json.loads(selected.to_json())}
    st.download_button("Download selection JSON", json.dumps(snapshot, indent=2),
                       file_name="airan_selection.json", mime="application/json")

st.caption("AI-RAN demo · Use the chart toolbar to save a figure.")

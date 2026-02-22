"""Smart-Support — Dash dashboard.

Run:
    uv run python frontend/dash_app.py
or via the FastAPI backend on port 8050 (standalone).
"""

from __future__ import annotations

import os

import requests
import dash
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
POLL_INTERVAL_MS = 4_000  # stats refresh

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg": "#060A11",
    "surface": "#0C1220",
    "card": "#101827",
    "border": "#1A2840",
    "accent": "#FF5E1A",
    "cyan": "#00CFEE",
    "teal": "#1FD4A4",
    "amber": "#FFB21E",
    "crimson": "#F03060",
    "text": "#DCE4F0",
    "muted": "#6A82A0",
}

CAT_COLORS = {
    "Billing": C["accent"],
    "Technical": C["cyan"],
    "Legal": C["amber"],
}

# ── Plotly base layout ────────────────────────────────────────────────────────
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'Space Mono', monospace", color=C["text"], size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    colorway=[C["accent"], C["cyan"], C["amber"], C["teal"], C["crimson"]],
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Barlow:wght@300;400;500;600;700&family=Bebas+Neue&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

body{
  background:#060A11 !important;
  color:#DCE4F0 !important;
  font-family:'Barlow',sans-serif !important;
  font-size:14px;
  min-height:100vh;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#060A11}
::-webkit-scrollbar-thumb{background:#1A2840;border-radius:2px}
::-webkit-scrollbar-thumb:hover{background:#FF5E1A}

/* ── Topbar ── */
.ss-topbar{
  position:sticky;top:0;z-index:200;
  background:rgba(6,10,17,0.92);
  backdrop-filter:blur(12px);
  border-bottom:1px solid #1A2840;
  padding:14px 36px;
  display:flex;align-items:center;gap:20px;
}
.ss-logo{
  font-family:'Bebas Neue',cursive;
  font-size:26px;letter-spacing:4px;color:#DCE4F0;
  line-height:1;
}
.ss-logo em{color:#FF5E1A;font-style:normal}
.ss-divider{width:1px;height:28px;background:#1A2840;flex-shrink:0}
.ss-tagline{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2.5px;
  text-transform:uppercase;color:#6A82A0;
}
.ss-spacer{flex:1}

/* status pills */
.pill{
  display:inline-flex;align-items:center;gap:6px;
  border-radius:20px;padding:4px 12px;
  font-family:'Space Mono',monospace;
  font-size:10px;font-weight:700;letter-spacing:.5px;
}
.pill-live{
  background:rgba(31,212,164,.1);
  border:1px solid rgba(31,212,164,.3);color:#1FD4A4;
}
.pill-model{
  background:rgba(0,207,238,.1);
  border:1px solid rgba(0,207,238,.25);color:#00CFEE;
}
.pill-cb-closed{color:#1FD4A4}
.pill-cb-open{color:#F03060}
.pill-cb-half_open{color:#FFB21E}
.pulse{
  width:6px;height:6px;border-radius:50%;background:#1FD4A4;
  animation:pulse 1.5s ease infinite;
}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.3;transform:scale(.8)}}
.uptime-txt{
  font-family:'Space Mono',monospace;
  font-size:10px;color:#6A82A0;
}

/* ── Main canvas ── */
.canvas{padding:28px 36px;max-width:1680px;margin:0 auto}
.gap{margin-bottom:20px}

/* ── Stat tile ── */
.stat-tile{
  background:#0C1220;
  border:1px solid #1A2840;
  border-radius:10px;
  padding:22px 24px;
  position:relative;overflow:hidden;
  transition:border-color .25s,transform .25s;
  cursor:default;
}
.stat-tile:hover{border-color:#FF5E1A;transform:translateY(-2px)}
.stat-tile::after{
  content:'';position:absolute;
  inset:0;background:linear-gradient(135deg,rgba(255,94,26,.04) 0%,transparent 60%);
  pointer-events:none;
}
.st-label{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2.5px;
  text-transform:uppercase;color:#6A82A0;
  margin-bottom:10px;
}
.st-value{
  font-family:'Bebas Neue',cursive;
  font-size:48px;line-height:1;
  letter-spacing:1px;
}
.st-value.orange{color:#FF5E1A}
.st-value.cyan{color:#00CFEE}
.st-value.teal{color:#1FD4A4}
.st-value.amber{color:#FFB21E}
.st-sub{font-size:11px;color:#6A82A0;margin-top:6px}
.st-accent-bar{
  position:absolute;bottom:0;left:0;right:0;height:2px;
}

/* ── Panel ── */
.panel{
  background:#0C1220;
  border:1px solid #1A2840;
  border-radius:10px;overflow:hidden;
  height:100%;
}
.panel-hdr{
  padding:13px 20px;
  border-bottom:1px solid #1A2840;
  background:rgba(26,40,64,.3);
  display:flex;align-items:center;gap:10px;
}
.panel-hdr-title{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2.5px;
  text-transform:uppercase;color:#DCE4F0;
  font-weight:700;
}
.panel-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.panel-dot.orange{background:#FF5E1A}
.panel-dot.cyan{background:#00CFEE}
.panel-dot.amber{background:#FFB21E}
.panel-dot.teal{background:#1FD4A4}
.panel-dot.crimson{background:#F03060}
.panel-body{padding:20px}

/* ── Form ── */
.f-label{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2px;
  text-transform:uppercase;color:#6A82A0;
  display:block;margin-bottom:6px;
}
textarea.f-inp,input.f-inp{
  width:100% !important;
  background:#060A11 !important;
  border:1px solid #1A2840 !important;
  border-radius:6px !important;
  color:#DCE4F0 !important;
  font-family:'Barlow',sans-serif !important;
  font-size:14px !important;
  padding:10px 14px !important;
  outline:none !important;
  resize:vertical;
  transition:border-color .2s,box-shadow .2s !important;
}
textarea.f-inp:focus,input.f-inp:focus{
  border-color:#FF5E1A !important;
  box-shadow:0 0 0 3px rgba(255,94,26,.12) !important;
}
textarea.f-inp::placeholder,input.f-inp::placeholder{
  color:#2D3F5A !important;
}

/* ── Buttons ── */
.btn-primary{
  background:#FF5E1A !important;
  border:none !important;
  border-radius:6px !important;
  color:#060A11 !important;
  font-family:'Barlow',sans-serif !important;
  font-weight:700 !important;
  font-size:13px !important;
  padding:10px 24px !important;
  cursor:pointer !important;
  transition:background .2s,transform .15s,box-shadow .2s !important;
  letter-spacing:.4px !important;
}
.btn-primary:hover{
  background:#ff7a40 !important;
  transform:translateY(-1px) !important;
  box-shadow:0 6px 20px rgba(255,94,26,.35) !important;
}
.btn-primary:active{transform:translateY(0) !important}
.btn-secondary{
  background:transparent !important;
  border:1px solid #00CFEE !important;
  border-radius:6px !important;
  color:#00CFEE !important;
  font-family:'Barlow',sans-serif !important;
  font-weight:600 !important;
  font-size:13px !important;
  padding:10px 24px !important;
  cursor:pointer !important;
  transition:background .2s,box-shadow .2s !important;
}
.btn-secondary:hover{
  background:rgba(0,207,238,.1) !important;
  box-shadow:0 0 18px rgba(0,207,238,.2) !important;
}
.btn-ghost{
  background:transparent !important;
  border:1px solid #1A2840 !important;
  border-radius:6px !important;
  color:#6A82A0 !important;
  font-family:'Barlow',sans-serif !important;
  font-size:13px !important;
  padding:10px 20px !important;
  cursor:pointer !important;
  transition:border-color .2s,color .2s !important;
}
.btn-ghost:hover{border-color:#6A82A0 !important;color:#DCE4F0 !important}

/* ── Result box ── */
.result-box{
  background:#060A11;
  border:1px solid #1A2840;
  border-radius:8px;
  padding:0;
  overflow:hidden;
  font-family:'Space Mono',monospace;
  font-size:12px;
}
.rb-row{
  display:flex;justify-content:space-between;align-items:center;
  padding:9px 16px;
  border-bottom:1px solid rgba(26,40,64,.6);
}
.rb-row:last-child{border-bottom:none}
.rb-key{color:#6A82A0}
.rb-val{color:#DCE4F0;font-weight:700}
.rb-val.orange{color:#FF5E1A}
.rb-val.cyan{color:#00CFEE}
.rb-val.teal{color:#1FD4A4}
.rb-val.amber{color:#FFB21E}
.rb-val.crimson{color:#F03060}
.rb-val.gray{color:#6A82A0}

/* ── Category chip ── */
.cat-chip{
  display:inline-block;
  padding:2px 10px;border-radius:20px;
  font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
}
.cat-billing{background:rgba(255,94,26,.14);color:#FF5E1A;border:1px solid rgba(255,94,26,.3)}
.cat-technical{background:rgba(0,207,238,.12);color:#00CFEE;border:1px solid rgba(0,207,238,.3)}
.cat-legal{background:rgba(255,178,30,.12);color:#FFB21E;border:1px solid rgba(255,178,30,.3)}

/* ── Urgency bar ── */
.urg-bar-bg{
  height:5px;background:#1A2840;border-radius:3px;
  overflow:hidden;margin-top:4px;
}
.urg-bar-fill{height:100%;border-radius:3px;transition:width .5s ease}

/* ── Agent bars ── */
.agent-row{display:flex;align-items:center;gap:12px;margin-bottom:14px}
.agent-row:last-child{margin-bottom:0}
.agent-name{
  font-family:'Space Mono',monospace;
  font-size:10px;color:#6A82A0;
  width:72px;flex-shrink:0;
}
.agent-track{flex:1;height:8px;background:#1A2840;border-radius:4px;overflow:hidden}
.agent-fill{
  height:100%;border-radius:4px;
  background:linear-gradient(90deg,#FF5E1A,#FFB21E);
  transition:width .4s ease;
}
.agent-cnt{
  font-family:'Space Mono',monospace;font-size:11px;
  color:#DCE4F0;width:36px;text-align:right;flex-shrink:0;
}

/* ── History table ── */
.hist-scroll{max-height:300px;overflow-y:auto}
.hist-tbl{width:100%;border-collapse:collapse}
.hist-tbl th{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:2px;text-transform:uppercase;
  color:#6A82A0;padding:8px 12px;
  border-bottom:1px solid #1A2840;text-align:left;
  position:sticky;top:0;background:#0C1220;
}
.hist-tbl td{
  padding:8px 12px;
  border-bottom:1px solid rgba(26,40,64,.5);
  color:#DCE4F0;font-size:12px;vertical-align:middle;
}
.hist-tbl tr:last-child td{border-bottom:none}
.hist-tbl tr:hover td{background:rgba(255,94,26,.04)}

/* ── Incident row ── */
.inc-row{
  padding:12px 16px;margin-bottom:8px;
  border:1px solid #1A2840;border-radius:8px;
  background:rgba(6,10,17,.7);
}
.inc-row:last-child{margin-bottom:0}
.inc-id{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:1px;color:#F03060;
  margin-bottom:4px;
}
.inc-text{font-size:12px;color:#6A82A0;margin-bottom:6px;line-height:1.4}
.inc-badge{
  display:inline-block;
  background:rgba(240,48,96,.12);color:#F03060;
  border:1px solid rgba(240,48,96,.3);
  border-radius:4px;padding:2px 8px;
  font-family:'Space Mono',monospace;font-size:10px;
}

/* ── Async poll ── */
.job-id-box{
  background:#060A11;border:1px dashed #1A2840;
  border-radius:6px;padding:10px 14px;
  font-family:'Space Mono',monospace;font-size:12px;color:#00CFEE;
  word-break:break-all;
}

/* ── Empty / error states ── */
.empty-state{
  text-align:center;padding:32px;
  font-family:'Space Mono',monospace;
  font-size:11px;color:#2D3F5A;
  letter-spacing:.5px;
}
.err-txt{
  font-family:'Space Mono',monospace;
  font-size:11px;color:#F03060;padding:10px;
}

/* ── Section heading ── */
.section-lbl{
  font-family:'Space Mono',monospace;
  font-size:9px;letter-spacing:3px;
  text-transform:uppercase;color:#6A82A0;
  margin-bottom:12px;display:flex;align-items:center;gap:10px;
}
.section-lbl::after{
  content:'';flex:1;height:1px;background:#1A2840;
}
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get(path: str, timeout: float = 4.0) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _post(path: str, payload: dict, timeout: float = 10.0) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def _urgency_color(score: float) -> str:
    if score < 0.4:
        return C["teal"]
    if score < 0.7:
        return C["amber"]
    return C["crimson"]


def _cat_class(cat: str) -> str:
    return f"cat-{cat.lower()}" if cat else ""


def _make_category_chart(counts: dict) -> go.Figure:
    cats = list(counts.keys())
    vals = [counts[c] for c in cats]
    colors = [CAT_COLORS.get(c, C["muted"]) for c in cats]
    fig = go.Figure(
        go.Bar(
            x=vals, y=cats, orientation="h",
            marker=dict(color=colors, opacity=0.88),
            text=vals, textposition="outside",
            textfont=dict(family="'Space Mono',monospace", size=11, color=C["text"]),
            hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        **_PLOTLY_BASE,
        height=150,
        bargap=0.38,
        xaxis=dict(
            showgrid=True, gridcolor="#1A2840", gridwidth=1,
            zeroline=False,
            tickfont=dict(size=9, color=C["muted"]),
        ),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11)),
    )
    return fig


def _render_result_box(r: dict) -> html.Div:
    """Render a routing result dict as a styled box."""
    if not r:
        return html.Div(className="empty-state", children="— awaiting ticket —")

    if "_error" in r:
        return html.Div(className="err-txt", children=f"Error: {r['_error']}")

    cat = r.get("category", "—")
    urgency_lbl = r.get("urgency", "—")
    score = float(r.get("urgency_score", 0))
    model = r.get("model_used", "—")
    agent = r.get("agent") or "—"
    dedup = r.get("dedup") or {}
    is_dup = dedup.get("is_duplicate", False)
    sim = dedup.get("similarity", 0.0)
    master = dedup.get("master_incident_id") or "—"

    score_pct = f"{score * 100:.1f}%"
    bar_color = _urgency_color(score)
    bar_w = f"{min(score * 100, 100):.0f}%"

    return html.Div(className="result-box", children=[
        html.Div(className="rb-row", children=[
            html.Span("CATEGORY", className="rb-key"),
            html.Span(cat, className=f"rb-val {_cat_class(cat)}"),
        ]),
        html.Div(className="rb-row", children=[
            html.Span("URGENCY", className="rb-key"),
            html.Div([
                html.Span(urgency_lbl, className="rb-val amber"),
                html.Div(className="urg-bar-bg", children=[
                    html.Div(
                        className="urg-bar-fill",
                        style={"width": bar_w, "background": bar_color},
                    )
                ]),
            ])
        ]),
        html.Div(className="rb-row", children=[
            html.Span("SCORE", className="rb-key"),
            html.Span(score_pct, className="rb-val", style={"color": bar_color}),
        ]),
        html.Div(className="rb-row", children=[
            html.Span("MODEL", className="rb-key"),
            html.Span(model.upper(), className="rb-val cyan"),
        ]),
        html.Div(className="rb-row", children=[
            html.Span("AGENT", className="rb-key"),
            html.Span(agent, className="rb-val teal"),
        ]),
        html.Div(className="rb-row", children=[
            html.Span("DUPLICATE", className="rb-key"),
            html.Span(
                "YES" if is_dup else "NO",
                className=f"rb-val {'crimson' if is_dup else 'gray'}",
            ),
        ]),
        *([] if not is_dup else [
            html.Div(className="rb-row", children=[
                html.Span("SIMILARITY", className="rb-key"),
                html.Span(f"{float(sim)*100:.1f}%", className="rb-val amber"),
            ]),
            html.Div(className="rb-row", children=[
                html.Span("MASTER", className="rb-key"),
                html.Span(str(master)[:20], className="rb-val crimson"),
            ]),
        ]),
    ])


def _render_agent_bars(agents: dict) -> list:
    if not agents:
        return [html.Div(className="empty-state", children="No agent data")]
    rows = []
    for name, info in agents.items():
        cap = info.get("capacity", 10)
        load = info.get("load", 0)
        pct = (load / cap * 100) if cap else 0
        rows.append(html.Div(className="agent-row", children=[
            html.Span(name, className="agent-name"),
            html.Div(className="agent-track", children=[
                html.Div(
                    className="agent-fill",
                    style={"width": f"{min(pct, 100):.0f}%"},
                )
            ]),
            html.Span(f"{load}/{cap}", className="agent-cnt"),
        ]))
    return rows


def _render_incidents(incidents: list) -> list:
    if not incidents:
        return [html.Div(className="empty-state", children="No master incidents")]
    rows = []
    for inc in incidents[:8]:
        rows.append(html.Div(className="inc-row", children=[
            html.Div(f"INCIDENT · {inc.get('incident_id','')[:16]}", className="inc-id"),
            html.Div(
                str(inc.get("representative_text", ""))[:100] + "…",
                className="inc-text",
            ),
            html.Span(
                f"{inc.get('ticket_count', 0)} tickets",
                className="inc-badge",
            ),
        ]))
    return rows


def _render_history_rows(history: list) -> list:
    if not history:
        return [html.Tr(html.Td(
            "No routes yet", colSpan=5,
            style={"textAlign": "center", "padding": "24px", "color": C["muted"]},
        ))]
    rows = []
    for entry in reversed(history[-20:]):
        cat = entry.get("category", "—")
        score = float(entry.get("urgency_score", 0))
        bar_color = _urgency_color(score)
        rows.append(html.Tr([
            html.Td(
                html.Span(cat, className=f"cat-chip {_cat_class(cat)}"),
            ),
            html.Td(
                entry.get("urgency", "—"),
                style={"color": C["amber"], "fontFamily": "'Space Mono',monospace", "fontSize": "11px"},
            ),
            html.Td(
                f"{score*100:.1f}%",
                style={"color": bar_color, "fontFamily": "'Space Mono',monospace", "fontSize": "11px"},
            ),
            html.Td(
                entry.get("agent") or "—",
                style={"color": C["teal"], "fontFamily": "'Space Mono',monospace", "fontSize": "11px"},
            ),
            html.Td(
                entry.get("model_used", "—").upper(),
                style={"color": C["cyan"], "fontFamily": "'Space Mono',monospace", "fontSize": "10px"},
            ),
        ]))
    return rows

# ── Layout ────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Smart-Support",
    update_title=None,
    suppress_callback_exceptions=True,
)

app.index_string = f"""<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>{{%title%}}</title>
  {{%favicon%}}
  {{%css%}}
  <style>{_CSS}</style>
</head>
<body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

_stat_tile = lambda label, value_id, color_cls, sub_text, bar_color: html.Div(
    className="stat-tile",
    children=[
        html.Div(label, className="st-label"),
        html.Div(id=value_id, children="—", className=f"st-value {color_cls}"),
        html.Div(sub_text, className="st-sub"),
        html.Div(className="st-accent-bar", style={"background": bar_color}),
    ],
)

app.layout = html.Div([
    dcc.Store(id="routing-history", data=[]),
    dcc.Store(id="async-job-id", data=None),
    dcc.Interval(id="stats-interval", interval=POLL_INTERVAL_MS, n_intervals=0),

    # ── Topbar ──
    html.Header(className="ss-topbar", children=[
        html.Div([
            html.H1("SMART<em>SUPPORT</em>", className="ss-logo"),
            html.P("AI Ticket Routing System", className="ss-tagline"),
        ]),
        html.Div(className="ss-divider"),
        html.Div(id="model-pill"),
        html.Div(id="cb-pill"),
        html.Div(className="ss-spacer"),
        html.Div(id="uptime-display", className="uptime-txt"),
        html.Div(className="ss-divider"),
        html.Div(className="pill pill-live", children=[
            html.Div(className="pulse"),
            "LIVE",
        ]),
    ]),

    # ── Main canvas ──
    html.Div(className="canvas", children=[

        # ── Row 1: Stat tiles ──
        dbc.Row([
            dbc.Col(_stat_tile("TICKETS ROUTED", "stat-total", "orange", "all time", "linear-gradient(90deg,#FF5E1A,#FFB21E)"), md=3, className="gap"),
            dbc.Col(_stat_tile("URGENT", "stat-urgent", "amber", "high priority", "#FFB21E"), md=3, className="gap"),
            dbc.Col(_stat_tile("WEBHOOK FIRES", "stat-webhooks", "teal", "alerts sent", "#1FD4A4"), md=3, className="gap"),
            dbc.Col(_stat_tile("MASTER INCIDENTS", "stat-incidents", "cyan", "dedup groups", "#00CFEE"), md=3, className="gap"),
        ], className="gap"),

        # ── Row 2: Routing form + Category chart ──
        dbc.Row([
            dbc.Col([
                html.Div(className="panel", children=[
                    html.Div(className="panel-hdr", children=[
                        html.Div(className="panel-dot orange"),
                        html.Div("Route a Ticket", className="panel-hdr-title"),
                    ]),
                    html.Div(className="panel-body", children=[
                        html.Label("Subject", className="f-label"),
                        dcc.Input(
                            id="inp-subject", type="text",
                            placeholder="e.g. Invoice charged twice",
                            className="f-inp",
                            style={"marginBottom": "14px"},
                            debounce=False,
                        ),
                        html.Label("Body", className="f-label"),
                        dcc.Textarea(
                            id="inp-body",
                            placeholder="Describe the issue in detail…",
                            className="f-inp",
                            style={"height": "90px", "marginBottom": "16px"},
                        ),
                        html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}, children=[
                            html.Button("▶ ROUTE", id="btn-route", className="btn-primary btn"),
                            html.Button("⏳ ASYNC", id="btn-async", className="btn-secondary btn"),
                            html.Button("✕ CLEAR", id="btn-clear", className="btn-ghost btn", n_clicks=0),
                        ]),
                        html.Div(style={"marginTop": "18px"}),
                        html.Div(id="route-result"),
                        html.Div(id="async-section", style={"marginTop": "12px"}),
                    ]),
                ])
            ], md=5, className="gap"),

            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Div(className="panel", style={"marginBottom": "20px"}, children=[
                            html.Div(className="panel-hdr", children=[
                                html.Div(className="panel-dot cyan"),
                                html.Div("Category Distribution", className="panel-hdr-title"),
                            ]),
                            html.Div(children=[
                                dcc.Graph(
                                    id="category-chart",
                                    config={"displayModeBar": False},
                                    style={"height": "150px"},
                                )
                            ]),
                        ]),
                    ], md=12),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Div(className="panel", children=[
                            html.Div(className="panel-hdr", children=[
                                html.Div(className="panel-dot orange"),
                                html.Div("Agent Load", className="panel-hdr-title"),
                            ]),
                            html.Div(className="panel-body", id="agent-bars",
                                    children=[html.Div(className="empty-state", children="Waiting for data…")]),
                        ])
                    ], md=12),
                ]),
            ], md=4, className="gap"),

            dbc.Col([
                html.Div(className="panel", style={"height": "100%"}, children=[
                    html.Div(className="panel-hdr", children=[
                        html.Div(className="panel-dot crimson"),
                        html.Div("Master Incidents", className="panel-hdr-title"),
                    ]),
                    html.Div(
                        className="panel-body",
                        id="incidents-panel",
                        style={"maxHeight": "400px", "overflowY": "auto"},
                        children=[html.Div(className="empty-state", children="No incidents detected")],
                    ),
                ]),
            ], md=3, className="gap"),
        ]),

        # ── Row 3: Routing history ──
        dbc.Row([
            dbc.Col([
                html.Div(className="panel", children=[
                    html.Div(className="panel-hdr", children=[
                        html.Div(className="panel-dot teal"),
                        html.Div("Routing History", className="panel-hdr-title"),
                        html.Div(className="ss-spacer"),
                        html.Span(id="history-count", style={
                            "fontFamily": "'Space Mono',monospace",
                            "fontSize": "9px", "color": C["muted"],
                        }),
                    ]),
                    html.Div(className="panel-body", style={"padding": "0"}, children=[
                        html.Div(className="hist-scroll", children=[
                            html.Table(className="hist-tbl", children=[
                                html.Thead(html.Tr([
                                    html.Th("Category"),
                                    html.Th("Urgency"),
                                    html.Th("Score"),
                                    html.Th("Agent"),
                                    html.Th("Model"),
                                ])),
                                html.Tbody(id="history-body"),
                            ])
                        ])
                    ]),
                ])
            ], md=12, className="gap"),
        ]),

        # ── Footer ──
        html.Div(
            style={"textAlign": "center", "padding": "20px 0 8px",
                    "borderTop": f"1px solid {C['border']}"},
            children=[
                html.Span(
                    "SMART-SUPPORT · AI TICKET ROUTING · BILINGUAL EN/DE",
                    style={
                        "fontFamily": "'Space Mono',monospace",
                        "fontSize": "9px", "letterSpacing": "2px",
                        "color": C["muted"], "textTransform": "uppercase",
                    },
                )
            ],
        ),
    ]),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("stat-total", "children"),
    Output("stat-urgent", "children"),
    Output("stat-webhooks", "children"),
    Output("stat-incidents", "children"),
    Output("category-chart", "figure"),
    Output("agent-bars", "children"),
    Output("incidents-panel", "children"),
    Output("model-pill", "children"),
    Output("cb-pill", "children"),
    Output("uptime-display", "children"),
    Input("stats-interval", "n_intervals"),
)
def refresh_stats(n):
    stats = _get("/stats")
    health = _get("/health")
    incidents_data = _get("/incidents")

    # ── Stat numbers ──
    total = stats.get("tickets_routed", 0) if stats else "—"
    urgent = stats.get("urgent_count", 0) if stats else "—"
    webhooks = stats.get("webhook_fires", 0) if stats else "—"
    incidents_count = stats.get("master_incidents", 0) if stats else "—"

    # ── Category chart ──
    counts = stats.get("category_counts", {"Billing": 0, "Technical": 0, "Legal": 0}) if stats else {"Billing": 0, "Technical": 0, "Legal": 0}
    fig = _make_category_chart(counts)

    # ── Agent bars ──
    agent_data = stats.get("agent_status") if stats else None
    if not agent_data and health:
        agent_data = None
    agent_bars = _render_agent_bars(agent_data or {})

    # ── Incidents ──
    inc_list = incidents_data.get("incidents", []) if incidents_data else []
    inc_children = _render_incidents(inc_list)

    # ── Pills ──
    variant = health.get("model_variant", "?").upper() if health else "OFFLINE"
    model_pill = html.Div(className="pill pill-model", children=[
        f"MODEL · {variant}"
    ])

    cb_state = health.get("circuit_breaker", "N/A") if health else "N/A"
    cb_cls = f"pill-cb-{cb_state.lower().replace(' ', '_')}" if health else "pill-cb-open"
    cb_pill = html.Div(
        f"CB · {cb_state.upper()}",
        style={
            "fontFamily": "'Space Mono',monospace",
            "fontSize": "10px", "letterSpacing": ".5px",
        },
        className=cb_cls,
    )

    # ── Uptime ──
    uptime_s = health.get("uptime_seconds", 0) if health else 0
    h, rem = divmod(int(uptime_s), 3600)
    m, s = divmod(rem, 60)
    uptime_txt = f"UPTIME  {h:02d}:{m:02d}:{s:02d}" if health else "API OFFLINE"

    return (
        str(total), str(urgent), str(webhooks), str(incidents_count),
        fig, agent_bars, inc_children,
        model_pill, cb_pill, uptime_txt,
    )


@callback(
    Output("route-result", "children"),
    Output("routing-history", "data"),
    Input("btn-route", "n_clicks"),
    State("inp-subject", "value"),
    State("inp-body", "value"),
    State("routing-history", "data"),
    prevent_initial_call=True,
)
def handle_route(n, subject, body, history):
    if not subject or not body:
        return html.Div("⚠ Subject and body are required.", className="err-txt"), history

    result = _post("/route", {"subject": subject, "body": body})
    new_history = (history or []) + [result]
    return _render_result_box(result), new_history


@callback(
    Output("async-section", "children"),
    Output("async-job-id", "data"),
    Input("btn-async", "n_clicks"),
    State("inp-subject", "value"),
    State("inp-body", "value"),
    prevent_initial_call=True,
)
def handle_async(n, subject, body):
    if not subject or not body:
        return html.Div("⚠ Subject and body are required.", className="err-txt"), None

    result = _post("/route/async", {"subject": subject, "body": body})
    if not result or "_error" in result:
        err = result.get("_error", "Unknown error") if result else "API offline"
        return html.Div(f"Error: {err}", className="err-txt"), None

    job_id = result.get("job_id", "")
    return html.Div([
        html.Div(style={"marginBottom": "8px", "fontFamily": "'Space Mono',monospace",
                        "fontSize": "10px", "color": C["muted"], "letterSpacing": "1px"},
                children="ASYNC JOB ACCEPTED"),
        html.Div(job_id, className="job-id-box"),
        html.Button(
            "POLL RESULT", id="btn-poll", className="btn-secondary btn",
            style={"marginTop": "10px"},
        ),
        html.Div(id="poll-result", style={"marginTop": "12px"}),
    ]), job_id


@callback(
    Output("poll-result", "children"),
    Input("btn-poll", "n_clicks"),
    State("async-job-id", "data"),
    prevent_initial_call=True,
)
def handle_poll(n, job_id):
    if not job_id:
        return html.Div("No job to poll.", className="err-txt")
    data = _get(f"/route/async/{job_id}")
    if not data:
        return html.Div("Could not reach API.", className="err-txt")
    status = data.get("status", "unknown")
    if status == "done":
        return _render_result_box(data.get("result") or {})
    return html.Div(
        f"Status: {status.upper()} — try again shortly",
        style={"fontFamily": "'Space Mono',monospace", "fontSize": "11px", "color": C["amber"]},
    )


@callback(
    Output("inp-subject", "value"),
    Output("inp-body", "value"),
    Output("route-result", "children", allow_duplicate=True),
    Output("async-section", "children", allow_duplicate=True),
    Input("btn-clear", "n_clicks"),
    prevent_initial_call=True,
)
def handle_clear(n):
    return "", "", html.Div(className="empty-state", children="— awaiting ticket —"), html.Div()


@callback(
    Output("history-body", "children"),
    Output("history-count", "children"),
    Input("routing-history", "data"),
)
def update_history_table(history):
    count = len(history or [])
    return _render_history_rows(history or []), f"{count} route{'s' if count != 1 else ''}"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Smart-Support Dashboard")
    print("  ━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  URL  → http://localhost:8050")
    print(f"  API  → {API_BASE}")
    print("  Make sure the FastAPI server is running first.\n")
    app.run(debug=False, host="0.0.0.0", port=8050)

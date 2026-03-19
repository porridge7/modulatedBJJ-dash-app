import numpy as np
import pickle
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

# --- Load data ---
with open("poincare_data_nested.pkl", "rb") as f:
    poincare_data = pickle.load(f)

# --- Extract parameter grids ---
Lambda_values = sorted({key[0] for key in poincare_data})
eps_values = sorted({key[1] for key in poincare_data})
Omega_values = sorted({key[2] for key in poincare_data})

# --- Extract initial conditions ---
sample_key = list(poincare_data.keys())[0]
initial_conditions = list(poincare_data[sample_key].keys())

# --- Consistent color mapping ---
color_map = {
    ic: f"hsl({i*360/len(initial_conditions)}, 80%, 50%)"
    for i, ic in enumerate(initial_conditions)
}

# --- Coordinate transform ---
def to_xyz(z, phi, R=1.0):
    x = np.sqrt(np.maximum(R**2 - z**2, 0)) * np.cos(phi)
    y = np.sqrt(np.maximum(R**2 - z**2, 0)) * np.sin(phi)
    return x, y, z

# --- Create sphere background ---
def make_sphere():
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    u, v = np.meshgrid(u, v)

    x = np.sin(v)*np.cos(u)
    y = np.sin(v)*np.sin(u)
    z = np.cos(v)

    return go.Surface(x=x, y=y, z=z, opacity=0.1, showscale=False)

# --- Dash app ---
app = Dash(__name__)

app.layout = html.Div([
    html.H3("Floquet-driven BJJ Poincaré Sections"),
    dcc.Markdown(r"""
                The time dependent Hamiltonian for the Floquet-driven system is:
                $$
                 H(z, \phi, t) = \frac{\Lambda}{2}z^2 - \sqrt{1-z^2}\cos(\phi)\left(1+\epsilon\sin(\Omega t)\right)
                $$
                We introduce the angle coordinate $\theta = \Omega t$ and write the extended Hamiltonian as
                 $$
                 H_\text{ext}(z, \phi, \theta, P_\theta) = H(z, \phi, \theta) + \Omega P_\theta
                 $$
                 We then use an implicit midpoint integrator which is symplectic and second order.
                 The implicit equations are solved by iterative method using 10 iterations.
    """,mathjax=True, style={
        "maxWidth": "800px",
        "margin": "auto",
        "textAlign": "left"
    }),
    html.Label("Lambda"),
    dcc.Slider(
        id="lambda-slider",
        min=0, max=len(Lambda_values)-1, step=1,
        value=0,
        marks={i: f"{v:.2f}" for i, v in enumerate(Lambda_values)}
    ),

    html.Label("epsilon"),
    dcc.Slider(
        id="eps-slider",
        min=0, max=len(eps_values)-1, step=1,
        value=0,
        marks={i: f"{v:.2f}" for i, v in enumerate(eps_values)}
    ),

    html.Label("Omega"),
    dcc.Slider(
        id="omega-slider",
        min=0, max=len(Omega_values)-1, step=1,
        value=0,
        marks={i: f"{v:.2f}" for i, v in enumerate(Omega_values)}
    ),

    dcc.Graph(id="sphere-plot")
])

# --- Callback ---
@app.callback(
    Output("sphere-plot", "figure"),
    Input("lambda-slider", "value"),
    Input("eps-slider", "value"),
    Input("omega-slider", "value")
)
def update_plot(i_L, i_eps, i_Om):

    Lambda = Lambda_values[i_L]
    eps = eps_values[i_eps]
    Omega = Omega_values[i_Om]

    key = (Lambda, eps, Omega)

    traces = []

    if key in poincare_data:
        for ic in initial_conditions:
            if ic not in poincare_data[key]:
                continue

            z_pts, phi_pts = poincare_data[key][ic]
            x, y, z = to_xyz(z_pts, phi_pts)

            traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=3, color=color_map[ic]),
                name=f"IC={ic}",
                hovertemplate=(
                    f"IC={ic}<br>"
                    f"z=%{{z:.3f}}<extra></extra>"
                )
            ))

    # add sphere
    traces.append(make_sphere())

    fig = go.Figure(data=traces)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig

# --- Run ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)

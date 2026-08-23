import streamlit as st
import numpy as np
import plotly.graph_objects as go
# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Streamlit Projectile Simulator",
    page_icon="🚀",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/Anonymous-25/Projectile_motion_streamlit/issues',
        'Report a bug': 'https://github.com/Anonymous-25/Projectile_motion_streamlit/issues/new',
        'About': '''
        ### :material/rocket_launch: Projectile Motion Simulator
        This interactive simulator models 2D projectile kinematics using **Semi-Implicit Euler integration**.
        
        * **Features:** Variable gravity, air resistance, dynamic atmosphere density, wind speed vectors, and custom simulation limits.
        * **Built with:** Streamlit, NumPy, and Plotly.
        '''
    }
)
# ============================================================
# TITLE
# ============================================================
st.title(":green[:material/rocket_launch: Projectile Motion Simulator]")
# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown('# :red[:material/simulation: Simulation Controls]', unsafe_allow_html=True)
# ============================================================
# SPEED RANGE
# ============================================================
st.sidebar.subheader(":material/rocket_launch: Launch configuration")
min_speed = st.sidebar.number_input(
    "Minimum Speed (m/s)",
    value=10.0,
    step=1.0
)
max_speed = st.sidebar.number_input(
    "Maximum Speed (m/s)",
    min_value=float(min_speed + 0.1),
    value=float(max(100.0, min_speed + 0.1)),
    step=1.0
)
# ============================================================
# CURRENT SPEED
# ============================================================
default_speed = (min_speed + max_speed) / 2
speed = st.sidebar.slider(
    "Current Speed (m/s)",
    min_value=float(min_speed),
    max_value=float(max_speed),
    value=float(default_speed),
    step=0.1
)
# ============================================================
# ANGLE
# ============================================================
angle = st.sidebar.slider(
    "Launch Angle (°)",
    min_value=0.1,
    max_value=89.9,
    value=45.0,
    step=0.000001
)
# ============================================================
# OBJECT PARAMETERS
# ============================================================
st.sidebar.subheader(":material/circle: Object")
mass = st.sidebar.number_input(
    "Mass (kg)",
    value=1.0,
    step=0.1
)
area = st.sidebar.number_input(
    "Cross-sectional Area (m²)",
    value=0.01,
    step=0.001,
    format="%.6f"
)
drag_coefficient = st.sidebar.number_input(
    "Drag Coefficient (Cd)",
    min_value=0.0,
    value=0.47,
    step=0.01
)
# ============================================================
# HEIGHT
# ============================================================
st.sidebar.subheader(":material/edit_location_alt: Position")
object_height = st.sidebar.number_input(
    "Object Initial Height (m)",
    value=10.0,
    step=0.1
)
ground_height = st.sidebar.number_input(
    "Ground Height (m)",
    value=0.0,
    step=0.1
)
# ============================================================
# CHECK HEIGHT
# ============================================================
# if object_height < ground_height:
#     st.error("Object height cannot be lower than ground height.")
#     st.stop()
# ============================================================
# GRAVITY
# ============================================================
st.sidebar.subheader(":material/falling: Gravity")
gravity_mode = st.sidebar.selectbox(
    "Gravity",
    [
        "Earth",
        "Moon",
        "Mars",
        "Jupiter",
        "Custom"
    ]
)
gravity_values = {
    "Earth": 9.81,
    "Moon": 1.62,
    "Mars": 3.71,
    "Jupiter": 24.79
}
if gravity_mode == "Custom":
    g = st.sidebar.number_input(
        "Custom Gravity (m/s²)",
        value=9.81,
        step=0.01
    )
else:
    g = gravity_values[gravity_mode]
# ============================================================
# AIR RESISTANCE
# ============================================================
st.sidebar.subheader(":material/filter_drama: Atmosphere")
air_resistance = st.sidebar.checkbox(
    "Enable Air Resistance",
    value=False
)
# ============================================================
# AIR PARAMETERS
# ============================================================
if air_resistance:
    air_density = st.sidebar.number_input(
        "Air Density (kg/m³)",
        value=1.225,
        step=0.001
    )
    wind_speed = st.sidebar.number_input(
        "Wind Speed (m/s)",
        value=0.0,
        step=0.1
    )
else:
    air_density = 0.0
    wind_speed = 0.0
# ============================================================
# SIMULATION LIMITS
# ============================================================
st.sidebar.subheader(":material/timer: Simulation Limits")
max_time_limit = st.sidebar.number_input(
    "Max Simulation Time (s)",
    min_value=0.0,
    value=10000.0,
    step=100.0
)
max_steps_limit = st.sidebar.number_input(
    "Max Step Limit",
    value=5000000,
    step=100000
)
# Add this at the very end of your sidebar code
st.sidebar.markdown("---")
st.sidebar.markdown("### :material/info: App Info")
st.sidebar.caption("v2.1.0 | Built with Streamlit & Plotly")
st.sidebar.caption("By Dhruv Meghwal (@Anonymous-25)")
st.sidebar.markdown("[:material/globe: GitHub Repository](https://github.com/Anonymous-25/Projectile_motion_streamlit/)")
# ============================================================
# NUMERICAL PROJECTILE SIMULATION
# ============================================================
def simulate_projectile(
    speed, angle, mass, area, cd, gravity, initial_height, ground, use_drag, air_density, wind_speed, max_time, max_steps
):
    theta = np.radians(angle)
    vx = speed * np.cos(theta)
    vy = speed * np.sin(theta)
    x, y, t = 0.0, float(initial_height), 0.0
    dt = 0.002
    times, xs, ys, velocities, drag_forces = [], [], [], [], []
    step = 0
    # Uses the sidebar limit inputs directly
    while y >= ground and t < max_time and step < max_steps:
        relative_vx = vx - wind_speed
        relative_vy = vy
        relative_speed = np.hypot(relative_vx, relative_vy)
        if use_drag and relative_speed > 1e-6:
            drag_force = 0.5 * air_density * cd * area * (relative_speed ** 2)
            ax_drag = -drag_force * relative_vx / (mass * relative_speed)
            ay_drag = -drag_force * relative_vy / (mass * relative_speed)
        else:
            drag_force = 0.0
            ax_drag = 0.0
            ay_drag = 0.0
        ax = ax_drag
        ay = -gravity + ay_drag
        times.append(t)
        xs.append(x)
        ys.append(y)
        velocities.append(np.hypot(vx, vy))
        drag_forces.append(drag_force)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        t += dt
        step += 1
    if len(ys) >= 2 and ys[-1] < ground:
        y1, y2 = ys[-2], ys[-1]
        fraction = (ground - y1) / (y2 - y1) if y2 != y1 else 0.0
        xs[-1] = xs[-2] + fraction * (xs[-1] - xs[-2])
        ys[-1] = ground
        times[-1] = times[-2] + fraction * (times[-1] - times[-2])
        velocities[-1] = velocities[-2] + fraction * (velocities[-1] - velocities[-2])
    return np.array(times), np.array(xs), np.array(ys), np.array(velocities), np.array(drag_forces)
# ============================================================
# RUN SIMULATION & EXTRACT RESULTS
# ============================================================
(time_data, x_data, y_data, velocity_data, drag_data) = simulate_projectile(
    speed=speed,
    angle=angle,
    mass=mass,
    area=area,
    cd=drag_coefficient,
    gravity=g,
    initial_height=object_height,
    ground=ground_height,
    use_drag=air_resistance,
    air_density=air_density,
    wind_speed=wind_speed,
    max_time=max_time_limit,
    max_steps=max_steps_limit
)
current_range = x_data[-1]
current_max_height = np.max(y_data)
flight_time = time_data[-1]
final_velocity = velocity_data[-1]
maximum_drag_force = np.max(drag_data)
# Dynamic Scaling with 15% Padding
x_axis_limit = max(10.0, current_range * 1.15)
y_axis_limit = max(10.0, current_max_height * 1.15)
# ============================================================
# DISPLAY LAYOUT
# ============================================================
coll1, coll2 = st.columns([3, 2])
# ============================================================
# PLOTLY GRAPH
# ============================================================
with coll1:
    fig = go.Figure()
    # Ground line
    fig.add_trace(
        go.Scatter(
            x=[0, x_axis_limit],
            y=[ground_height, ground_height],
            mode="lines",
            name="Ground",
            line=dict(width=3, dash="dash")
        )
    )
    # Trajectory
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            name="Projectile",
            line=dict(width=4),
            hovertemplate=(
                "Distance: %{x:.2f} m<br>"
                "Height: %{y:.2f} m<extra></extra>"
            )
        )
    )
    # Launch Point
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[object_height],
            mode="markers",
            name="Launch Point",
            marker=dict(size=12,color="#00FF7F")
        )
    )
    # Landing Point
    fig.add_trace(
        go.Scatter(
            x=[current_range],
            y=[ground_height],
            mode="markers",
            name="Landing Point",
            marker=dict(size=12,color="#FF4B4B")
        )
    )
    # Graph Settings
    fig.update_layout(
        title=f"Projectile Motion — {speed:.1f} m/s at {angle:.1f}°",
        xaxis=dict(
            title="Horizontal Distance (m)",
            range=[0, x_axis_limit],
            
        ),
        yaxis=dict(
            title="Height (m)",
            range=[0, y_axis_limit],
            
            scaleanchor="x",
            scaleratio=1
        ),
        hovermode="closest",
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)
# ============================================================
# MAIN RESULTS
# ============================================================
with coll2:
    st.subheader(":material/analytics: Simulation Results")
    col1, col2_res = st.columns(2)
    with col1:
        st.metric("Range", f"{current_range:.2f} m")
    with col2_res:
        st.metric("Maximum Height", f"{current_max_height:.2f} m")
    with col1:
        st.metric("Flight Time", f"{flight_time:.2f} s")
    with col2_res:
        st.metric("Final Velocity", f"{final_velocity:.2f} m/s")
    if air_resistance:
        st.subheader(":material/air: Air Resistance")
        col1_air, col2_air = st.columns(2)
        with col1_air:
            st.metric("Maximum Drag Force", f"{maximum_drag_force:.3f} N")
        with col2_air:
            st.metric("Air Density", f"{air_density:.3f} kg/m³")
        with col1_air:
            st.metric("Drag Coefficient", f"{drag_coefficient:.2f}")
    st.subheader(":material/circle: Object & Environment")
    col1_env, col2_env = st.columns(2)
    with col1_env:
        st.metric("Mass", f"{mass:.3f} kg")
    with col2_env:
        st.metric("Gravity", f"{g:.2f} m/s²")
    with col1_env:
        st.metric("Initial Height", f"{object_height:.2f} m")
    with col2_env:
        st.metric("Ground Height", f"{ground_height:.2f} m")
# ============================================================
# PHYSICS EXPANDER
# ============================================================
with st.expander(":material/book: Physics Equations"):
    st.markdown("### Gravity")
    st.latex(r"F_g = mg")
    st.markdown("### Air Resistance")
    st.latex(r"F_d = \frac{1}{2}\rho C_d A v^2")
    st.markdown("### Drag Acceleration")
    st.latex(r"a_d = \frac{F_d}{m}")
    st.markdown("### Without Air Resistance")
    st.latex(r"a_x = 0")
    st.latex(r"a_y = -g")
    st.markdown("Mass therefore has no effect on the ideal trajectory.")
    st.markdown("### With Air Resistance")
    st.latex(r"a_x = -\frac{F_d}{m}\frac{v_x-v_{wind}}{v_{relative}}")
    st.latex(r"a_y = -g-\frac{F_d}{m}\frac{v_y}{v_{relative}}")
    if air_resistance:
        st.success(
            ":material/air: Air resistance is ON. "
            "Changing mass, area, drag coefficient, air density, "
            "or wind speed will change the trajectory."
        )
    else:
        st.info(
            ":material/lightbulb: Air resistance is OFF. "
            "Mass does not affect ideal projectile motion. "
            "Enable Air Resistance to see the effect of mass."
        )

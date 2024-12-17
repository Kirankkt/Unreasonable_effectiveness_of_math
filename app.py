import streamlit as st
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Unreasonable Effectiveness of Mathematics",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
intro_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json")
philosophy_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_puciaact.json")
math_theories_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_mjlh3hcy.json")
physics_animation = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_x62chJ.json")

# Sidebar Navigation
st.sidebar.title("Navigation")
sections = [
    "Introduction",
    "Philosophical Foundations",
    "Historical Mathematical Theories",
    "Mathematics in Physics",
    "Interactive Visualizations",
    "Quiz",
    "References",
]
selection = st.sidebar.radio("Go to", sections)

# Introduction Section
if selection == "Introduction":
    st.title("📐 The Unreasonable Effectiveness of Mathematics in the Natural Sciences")
    st_lottie(intro_animation, height=300, key="intro")
    st.markdown("""
    Eugene Wigner, in his seminal essay, explored the mysterious ability of mathematics to describe and predict phenomena in the physical world with astonishing precision. This app delves into the philosophical underpinnings, historical developments, and practical applications that highlight this "unreasonable effectiveness."
    """)
    st.markdown("""
    **Key Objectives:**
    - Understand Wigner's perspective on mathematics and physics.
    - Explore philosophical debates: Platonism vs. Nominalism.
    - Trace the history of mathematical theories preceding their physical applications.
    - Engage with interactive animations and visualizations.
    """)

# Philosophical Foundations Section
elif selection == "Philosophical Foundations":
    st.title("🧠 Philosophical Foundations")
    st_lottie(philosophy_animation, height=300, key="philosophy")
    
    st.header("Platonism")
    st.markdown("""
    **Platonism** posits that mathematical entities exist independently of human minds. According to this view, numbers, shapes, and other mathematical constructs are discovered, not invented. This perspective aligns with Wigner's observation, suggesting that mathematics has an intrinsic reality that nature taps into.
    """)
    
    st.header("Nominalism")
    st.markdown("""
    **Nominalism** argues that mathematical entities are merely symbolic representations without any independent existence. From this standpoint, mathematics is a creation of the human mind, developed as a language to describe patterns and relationships observed in the physical world.
    """)
    
    st.markdown("""
    **Debate:**
    The tension between Platonism and Nominalism centers on whether mathematics is a fundamental aspect of reality or a sophisticated tool crafted by humans. Wigner's reflections often lean towards Platonism, given the uncanny applicability of abstract mathematics in empirical sciences.
    """)

# Historical Mathematical Theories Section
elif selection == "Historical Mathematical Theories":
    st.title("📚 Historical Mathematical Theories")
    st_lottie(math_theories_animation, height=300, key="math_theories")
    
    st.header("1. Complex Numbers")
    st.markdown("""
    Initially developed to solve polynomial equations, **complex numbers** were met with skepticism. Today, they are indispensable in fields like quantum mechanics, electrical engineering, and fluid dynamics.
    """)
    st.subheader("Visualization: Complex Plane")
    complex_fig = go.Figure()
    complex_fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines+markers',
        name='Unit Complex Number',
        marker=dict(size=[10, 20]),
        line=dict(color='blue')
    ))
    complex_fig.update_layout(title="Complex Plane Representation", xaxis_title="Real", yaxis_title="Imaginary")
    st.plotly_chart(complex_fig, use_container_width=True)
    
    st.header("2. Non-Euclidean Geometry")
    st.markdown("""
    **Non-Euclidean geometries** challenged the long-held Euclidean axioms, paving the way for Einstein's theory of General Relativity, which describes the curvature of spacetime.
    """)
    
    st.header("3. Group Theory")
    st.markdown("""
    **Group theory**, abstracted from geometry, now underpins particle physics and the Standard Model, explaining fundamental symmetries in nature.
    """)

# Mathematics in Physics Section
elif selection == "Mathematics in Physics":
    st.title("🔬 Mathematics in Physics")
    st_lottie(physics_animation, height=300, key="physics")
    
    st.header("Quantum Mechanics")
    st.markdown("""
    The formulation of **quantum mechanics** relies heavily on linear algebra and complex Hilbert spaces. The mathematical framework predicts phenomena like superposition and entanglement, which have been experimentally verified.
    """)
    
    st.header("General Relativity")
    st.markdown("""
    **General Relativity** uses differential geometry to describe the fabric of spacetime. Einstein's field equations, which are highly mathematical, have been confirmed through observations like gravitational lensing and the recent detection of gravitational waves.
    """)
    
    st.header("String Theory")
    st.markdown("""
    **String Theory** employs advanced mathematical concepts from topology, algebraic geometry, and higher-dimensional spaces to attempt a unified description of all fundamental forces.
    """)

# Interactive Visualizations Section
elif selection == "Interactive Visualizations":
    st.title("🔍 Interactive Visualizations")
    
    st.header("1. Complex Number Operations")
    st.markdown("""
    Explore how complex numbers behave under addition and multiplication.
    """)
    # Interactive Plot
    import numpy as np
    real = np.linspace(-10, 10, 400)
    imag = np.linspace(-10, 10, 400)
    real_grid, imag_grid = np.meshgrid(real, imag)
    magnitude = np.sqrt(real_grid**2 + imag_grid**2)
    phase = np.arctan2(imag_grid, real_grid)
    
    fig1 = go.Figure(data=go.Contour(
        z=magnitude,
        x=real,
        y=imag,
        colorscale='Viridis',
        colorbar=dict(title='Magnitude')
    ))
    fig1.update_layout(title='Magnitude of Complex Numbers', xaxis_title='Real', yaxis_title='Imaginary')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.header("2. Visualization of Non-Euclidean Geometry")
    st.markdown("""
    Visual representation of curved spacetime as described by General Relativity.
    """)
    # Simple interactive 3D plot
    theta = np.linspace(0, 2*np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    r = 1  # Radius
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    fig2 = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale='Earth')])
    fig2.update_layout(title='3D Sphere Representing Curved Space', autosize=True)
    st.plotly_chart(fig2, use_container_width=True)

# Quiz Section
elif selection == "Quiz":
    st.title("📝 Quiz: Test Your Understanding")
    
    st.markdown("""
    **Question 1:** Which philosophical view posits that mathematical entities exist independently of human minds?
    """)
    q1 = st.radio("Select an option:", ("Platonism", "Nominalism", "Constructivism", "Empiricism"))
    if st.button("Submit Q1"):
        if q1 == "Platonism":
            st.success("Correct! Platonism asserts the independent existence of mathematical entities.")
        else:
            st.error("Incorrect. The correct answer is Platonism.")
    
    st.markdown("---")
    
    st.markdown("""
    **Question 2:** Which mathematical theory is essential for the formulation of quantum mechanics?
    """)
    q2 = st.radio("Select an option:", ("Calculus", "Complex Numbers", "Topology", "Graph Theory"), key="q2")
    if st.button("Submit Q2"):
        if q2 == "Complex Numbers":
            st.success("Correct! Complex numbers are fundamental in quantum mechanics.")
        else:
            st.error("Incorrect. The correct answer is Complex Numbers.")
    
    st.markdown("---")
    
    st.markdown("""
    **Question 3:** What does Non-Euclidean geometry primarily deal with?
    """)
    q3 = st.radio("Select an option:", ("Flat surfaces", "Curved surfaces", "Two-dimensional spaces", "None of the above"), key="q3")
    if st.button("Submit Q3"):
        if q3 == "Curved surfaces":
            st.success("Correct! Non-Euclidean geometry deals with curved surfaces.")
        else:
            st.error("Incorrect. The correct answer is Curved surfaces.")

# References Section
elif selection == "References":
    st.title("📚 References & Further Reading")
    st.markdown("""
    - **Wigner, E. P.** (1960). *The Unreasonable Effectiveness of Mathematics in the Natural Sciences*. Communications in Pure and Applied Mathematics, 13(1), 1-14.
    - **Polya, G.** (1975). *Mathematics and Plausible Reasoning*. Princeton University Press.
    - **Penrose, R.** (2004). *The Road to Reality: A Complete Guide to the Laws of the Universe*. Vintage.
    - **Lovelock, J.** (2006). *The Dream of a Final Theory*. Houghton Mifflin Harcourt.
    - **Lange, D.** (1993). *Physics and Philosophy: The Revolution in Modern Science*. Prometheus Books.
    - **Linnebo, Ø.** (2004). *Scientific Realism and Structural Realism*. Oxford University Press.
    - **Strogatz, S.** (2009). *The Joy of x: A Guided Tour of Math, from One to Infinity*. Houghton Mifflin Harcourt.
    """)
    st.markdown("""
    **Online Resources:**
    - [LottieFiles - Free Animations](https://lottiefiles.com/)
    - [Plotly - Interactive Graphing](https://plotly.com/python/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    """)

# Footer
st.markdown("---")
st.markdown("""
    **Developed by:** [Your Name]
    **Contact:** your.email@example.com
    **GitHub:** [github.com/yourusername](https://github.com/yourusername)
""")


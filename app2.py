import streamlit as st
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objs as go
import numpy as np

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
    "Mathematical Theories",
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

# Mathematical Theories Section
elif selection == "Mathematical Theories":
    st.title("📚 Mathematical Theories and Their Physical Applications")
    st_lottie(math_theories_animation, height=300, key="math_theories")
    
    # List of mathematical theories with descriptions
    math_concepts = [
        {
            "title": "1. Newtonian Mechanics and Calculus",
            "what": "Isaac Newton's invention of calculus allowed the laws of motion to be mathematically described. These mathematical principles precisely explain planetary motion, falling objects, and more.",
            "effectiveness": "Calculus predicts behaviors in the real world with astonishing accuracy.",
            "visualization_type": "projectile_motion",
        },
        {
            "title": "2. Maxwell’s Equations and Electromagnetism",
            "what": "James Clerk Maxwell unified electricity and magnetism into four elegant equations.",
            "effectiveness": "These abstract equations led to the understanding of light as an electromagnetic wave, radio waves, and more, profoundly shaping our understanding of the universe.",
            "visualization_type": "electromagnetic_wave",
        },
        {
            "title": "3. General Relativity and Non-Euclidean Geometry",
            "what": "Einstein used the mathematics of differential geometry (developed in abstract settings) to describe gravity as the curvature of spacetime.",
            "effectiveness": "The theory correctly predicts the bending of light, GPS time dilation, and black holes—phenomena unimagined when the math was developed.",
            "visualization_type": "gravitational_lensing",
        },
        {
            "title": "4. Quantum Mechanics and Linear Algebra",
            "what": "Quantum mechanics relies heavily on linear algebra, matrices, and Hilbert spaces to describe particle behavior.",
            "effectiveness": "This abstract math describes the atomic and subatomic world so precisely that it underpins technologies like semiconductors, lasers, and quantum computing.",
            "visualization_type": "quantum_superposition",
        },
        {
            "title": "5. The Standard Model and Group Theory",
            "what": "The Standard Model of particle physics relies on advanced group theory (symmetry operations) to describe fundamental particles and forces.",
            "effectiveness": "Symmetry principles predicted particles (like the Higgs boson) years before they were discovered experimentally.",
            "visualization_type": "higgs_boson",
        },
        {
            "title": "6. Fourier Analysis and Signal Processing",
            "what": "Fourier transforms (decomposing signals into sine waves) describe waves, vibrations, and signals in mathematics.",
            "effectiveness": "This concept enables modern telecommunications, image compression (JPEG), audio processing, and even quantum mechanics.",
            "visualization_type": "fourier_transform",
        },
        {
            "title": "7. Chaos Theory and Fractals",
            "what": "Chaos theory describes how small changes in initial conditions can lead to large-scale unpredictable outcomes (e.g., the butterfly effect).",
            "effectiveness": "Applications include weather systems, population dynamics, and fractals, which appear everywhere in nature (e.g., trees, coastlines).",
            "visualization_type": "fractal",
        },
        {
            "title": "8. Complex Numbers and Electrical Engineering",
            "what": "The imaginary unit i (where i² = −1) seems purely abstract.",
            "effectiveness": "Complex numbers are indispensable for solving problems in electrical circuits, signal processing, and quantum mechanics.",
            "visualization_type": "complex_plane",
        },
        {
            "title": "9. Statistics and Machine Learning",
            "what": "Statistical methods mathematically describe uncertainty, probability, and trends in data.",
            "effectiveness": "These tools now form the foundation of machine learning and AI, driving advancements in nearly every domain.",
            "visualization_type": "machine_learning",
        },
        {
            "title": "10. Euler’s Formula and Beauty",
            "what": "Euler’s formula, e^{iπ} + 1 = 0, links five of the most fundamental constants in mathematics: e, i, π, 1, and 0.",
            "effectiveness": "This single equation connects analysis, geometry, and algebra and reveals deep truths about the nature of mathematics itself.",
            "visualization_type": "euler_formula",
        },
        {
            "title": "11. Information Theory",
            "what": "Claude Shannon mathematically described the nature of information using entropy and coding theory.",
            "effectiveness": "It underpins everything from modern communication systems to compression algorithms and AI.",
            "visualization_type": "information_entropy",
        },
        {
            "title": "12. Prime Numbers and Cryptography",
            "what": "Prime numbers were considered mathematical curiosities.",
            "effectiveness": "They now form the backbone of modern encryption systems (e.g., RSA), securing data worldwide.",
            "visualization_type": "rsa_encryption",
        },
        {
            "title": "13. Game Theory in Biology and Economics",
            "what": "Game theory originated as a mathematical framework for decision-making.",
            "effectiveness": "It surprisingly models animal behaviors (e.g., evolutionary stable strategies) and economic markets with precision.",
            "visualization_type": "prisoners_dilemma",
        },
        {
            "title": "14. Statistical Mechanics and Thermodynamics",
            "what": "Boltzmann used probabilistic math to explain entropy and thermodynamic laws.",
            "effectiveness": "This theory successfully links the microscopic (atoms) to the macroscopic (temperature, pressure).",
            "visualization_type": "entropy",
        },
        {
            "title": "15. Topology in Physics",
            "what": "Topology studies properties of shapes that remain unchanged under deformation.",
            "effectiveness": "It now plays a crucial role in modern physics, such as understanding topological phases of matter (e.g., quantum Hall effect).",
            "visualization_type": "topological_insulators",
        },
        {
            "title": "16. Neural Networks and Backpropagation",
            "what": "Neural networks utilize relatively simple mathematical operations like matrix multiplication and derivatives.",
            "effectiveness": "These concepts, once theoretical, are now driving innovations in AI.",
            "visualization_type": "neural_network",
        },
        {
            "title": "17. Symmetry and the Universe (Noether's Theorem)",
            "what": "Emmy Noether proved that every symmetry in physics corresponds to a conserved quantity (e.g., time symmetry = energy conservation).",
            "effectiveness": "This deep connection between math and physics shapes modern theories like relativity and quantum field theory.",
            "visualization_type": "noether_theorem",
        },
        {
            "title": "18. Mathematical Biology",
            "what": "Differential equations and network models describe everything from predator-prey interactions (Lotka-Volterra) to infectious disease spread.",
            "effectiveness": "Math reveals deep insights into biology, like population growth, neuron firing, and cancer dynamics.",
            "visualization_type": "lotka_volterra",
        },
        {
            "title": "19. Deep Space Navigation and Math",
            "what": "Math-based physics allowed missions like Voyager 1 to navigate billions of miles with pinpoint accuracy.",
            "effectiveness": "Abstract theories guide spacecraft through gravity assists, proving the power of math far beyond Earth.",
            "visualization_type": "space_navigation",
        },
        {
            "title": "20. Quantum Field Theory and Renormalization",
            "what": "QFT merges quantum mechanics and special relativity using abstract math (like Feynman diagrams and renormalization).",
            "effectiveness": "It provides incredibly accurate predictions for particle interactions, like the electron’s magnetic moment.",
            "visualization_type": "feynman_diagram",
        },
    ]
    
    # Function to generate visualizations based on type
    def generate_visualization(v_type):
        if v_type == "projectile_motion":
            st.markdown("**Visualization: Projectile Motion Under Gravity**")
            t = np.linspace(0, 10, 400)
            g = 9.81  # Acceleration due to gravity
            y = 100 - 0.5 * g * t**2  # Height as a function of time
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Height'))
            fig.update_layout(title="Projectile Motion Under Gravity", xaxis_title="Time (s)", yaxis_title="Height (m)")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "electromagnetic_wave":
            st.markdown("**Visualization: Electromagnetic Wave Propagation**")
            x = np.linspace(0, 10, 400)
            t = 0
            E = np.sin(x - t)
            B = np.sin(x - t)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=E, mode='lines', name='Electric Field (E)', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x, y=B, mode='lines', name='Magnetic Field (B)', line=dict(color='red')))
            fig.update_layout(title="Electromagnetic Wave Propagation", xaxis_title="Position (x)", yaxis_title="Field Strength")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "gravitational_lensing":
            st.markdown("**Visualization: Gravitational Lensing Simulation**")
            theta = np.linspace(0, 2*np.pi, 400)
            r = 1 + 0.1 * np.sin(5 * theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Curved Spacetime'))
            fig.update_layout(title="Gravitational Lensing Simulation", xaxis_title="X-axis", yaxis_title="Y-axis", 
                              xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "quantum_superposition":
            st.markdown("**Visualization: Quantum Superposition States**")
            vectors = {
                'State 1': [1, 0],
                'State 2': [0, 1],
                'Superposition': [1/np.sqrt(2), 1/np.sqrt(2)]
            }
            
            fig = go.Figure()
            for state, vec in vectors.items():
                fig.add_trace(go.Scatter(
                    x=[0, vec[0]],
                    y=[0, vec[1]],
                    mode='lines+markers',
                    name=state,
                    marker=dict(size=[10, 12]),
                ))
            fig.update_layout(title="Quantum Superposition States", xaxis_title="Re", yaxis_title="Im", 
                              xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "higgs_boson":
            st.markdown("**Visualization: Higgs Boson Detection**")
            # Placeholder for Higgs Boson visualization
            st.image("https://upload.wikimedia.org/wikipedia/commons/4/4b/Higgs_boson.svg", caption="Higgs Boson Representation", use_column_width=True)
        
        elif v_type == "fourier_transform":
            st.markdown("**Visualization: Fourier Transform**")
            t = np.linspace(0, 1, 400)
            freq = 5
            signal = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=signal, mode='lines', name='Time Domain Signal'))
            fig.update_layout(title="Time Domain Signal", xaxis_title="Time (s)", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True)
            
            # Fourier Transform
            fft_vals = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(t), d=t[1]-t[0])
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=fft_freq[:200], y=np.abs(fft_vals)[:200], mode='lines', name='Frequency Domain'))
            fig2.update_layout(title="Frequency Domain Signal", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
            st.plotly_chart(fig2, use_container_width=True)
        
        elif v_type == "fractal":
            st.markdown("**Visualization: Mandelbrot Fractal**")
            # Simple Mandelbrot set visualization
            from PIL import Image
            import io
            
            def mandelbrot(x, y, max_iters):
                c0 = complex(x,y)
                c = 0
                for i in range(max_iters):
                    if abs(c) > 2:
                        return i
                    c = c * c + c0
                return max_iters

            width, height = 300, 200
            max_iters = 20
            img = Image.new('RGB', (width, height))
            pixels = img.load()
            for px in range(width):
                for py in range(height):
                    x = (px / width) * 3.5 - 2.5
                    y = (py / height) * 2.0 - 1.0
                    m = mandelbrot(x, y, max_iters)
                    color = 255 - int(m * 255 / max_iters)
                    pixels[px, py] = (color, color, color)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            st.image(buf, caption="Mandelbrot Fractal", use_column_width=True)
        
        elif v_type == "complex_plane":
            st.markdown("**Visualization: Complex Plane Representation**")
            real = np.linspace(-10, 10, 400)
            imag = np.linspace(-10, 10, 400)
            real_grid, imag_grid = np.meshgrid(real, imag)
            magnitude = np.sqrt(real_grid**2 + imag_grid**2)
            phase = np.arctan2(imag_grid, real_grid)
            
            fig = go.Figure(data=go.Contour(
                z=magnitude,
                x=real,
                y=imag,
                colorscale='Viridis',
                colorbar=dict(title='Magnitude')
            ))
            fig.update_layout(title="Magnitude of Complex Numbers", xaxis_title="Real", yaxis_title="Imaginary")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "machine_learning":
            st.markdown("**Visualization: Decision Boundary of a Simple Neural Network**")
            from sklearn.datasets import make_moons
            from sklearn.model_selection import train_test_split
            from sklearn.neural_network import MLPClassifier
            
            X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            
            # Create meshgrid
            h = .02  # step size in the mesh
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z, showscale=False, colorscale='RdBu', opacity=0.5))
            fig.add_trace(go.Scatter(x=X_train[:,0], y=X_train[:,1], mode='markers', marker=dict(color=y_train, colorscale='RdBu'), name='Train Data'))
            fig.add_trace(go.Scatter(x=X_test[:,0], y=X_test[:,1], mode='markers', marker=dict(color=y_test, symbol='x'), name='Test Data'))
            fig.update_layout(title="Decision Boundary of a Simple Neural Network", xaxis_title="Feature 1", yaxis_title="Feature 2")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "euler_formula":
            st.markdown("**Visualization: Euler’s Formula in the Complex Plane**")
            theta = np.linspace(0, 2*np.pi, 400)
            points = np.exp(1j * theta)
            x = points.real
            y = points.imag
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='e^{iθ}'))
            fig.update_layout(title="Euler’s Formula on the Complex Plane", xaxis_title="Re", yaxis_title="Im",
                              xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "information_entropy":
            st.markdown("**Visualization: Information Entropy**")
            probabilities = [0.1, 0.2, 0.3, 0.4]
            labels = ['A', 'B', 'C', 'D']
            entropy = -sum(p * np.log2(p) for p in probabilities)
            st.markdown(f"**Entropy:** {entropy:.2f} bits")
            
            fig = go.Figure(data=[go.Pie(labels=labels, values=probabilities, hole=.3)])
            fig.update_layout(title="Information Entropy Visualization")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "rsa_encryption":
            st.markdown("**Visualization: RSA Encryption Process**")
            st.markdown("""
            **Step 1:** Choose two prime numbers, \( p = 61 \) and \( q = 53 \).
            
            **Step 2:** Compute \( n = p \times q = 3233 \).
            
            **Step 3:** Compute Euler's totient function, \( \phi(n) = (p-1)(q-1) = 3120 \).
            
            **Step 4:** Choose an encryption key \( e = 17 \) (coprime with 3120).
            
            **Step 5:** Compute the decryption key \( d \) such that \( d \times e \mod \phi(n) = 1 \). Here, \( d = 2753 \).
            
            **Encryption:** \( \text{Cipher} = \text{Message}^e \mod n \)
            
            **Decryption:** \( \text{Message} = \text{Cipher}^d \mod n \)
            """)
            # Simple encryption/decryption example
            message = 123
            e = 17
            d = 2753
            n = 3233
            cipher = pow(message, e, n)
            decrypted = pow(cipher, d, n)
            st.markdown(f"**Original Message:** {message}")
            st.markdown(f"**Encrypted Cipher:** {cipher}")
            st.markdown(f"**Decrypted Message:** {decrypted}")
        
        elif v_type == "prisoners_dilemma":
            st.markdown("**Visualization: Prisoner's Dilemma Payoff Matrix**")
            payoff_matrix = {
                'Cooperate': {'Cooperate': (3, 3), 'Defect': (0, 5)},
                'Defect': {'Cooperate': (5, 0), 'Defect': (1, 1)}
            }
            st.table(payoff_matrix)
            st.markdown("""
            **Explanation:**
            - If both players cooperate, they each receive a moderate reward.
            - If one defects while the other cooperates, the defector gets a high reward while the cooperator gets nothing.
            - If both defect, they both receive a low reward.
            """)
        
        elif v_type == "entropy":
            st.markdown("**Visualization: Entropy in Thermodynamics**")
            st.markdown("""
            **Definition:** Entropy is a measure of the disorder or randomness in a system.
            
            **Visualization:**
            """)
            t = np.linspace(0, 10, 400)
            S = np.log(t + 1)  # Simplified entropy increase
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Entropy'))
            fig.update_layout(title="Entropy Over Time", xaxis_title="Time (s)", yaxis_title="Entropy (S)")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "topological_insulators":
            st.markdown("**Visualization: Topological Insulators**")
            st.markdown("""
            **Description:** Topological insulators are materials that conduct electricity on their surface but not in their bulk. This property arises from the material's topological properties.
            
            **Visualization:**
            """)
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Topological_insulator.svg/1200px-Topological_insulator.svg.png", caption="Topological Insulator Structure", use_column_width=True)
        
        elif v_type == "neural_network":
            st.markdown("**Visualization: Simple Neural Network Architecture**")
            st.image("https://upload.wikimedia.org/wikipedia/commons/8/86/Artificial_neural_network.svg", caption="Neural Network Structure", use_column_width=True)
            st.markdown("""
            **Backpropagation:** The process of training a neural network by adjusting weights based on the error rate obtained in the previous epoch.
            """)
        
        elif v_type == "noether_theorem":
            st.markdown("**Visualization: Noether's Theorem and Conservation Laws**")
            st.markdown("""
            **Example:** Time Symmetry ↔ Conservation of Energy
            
            **Visualization:**
            """)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines+markers', name='Symmetry'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines+markers', name='Conserved Quantity'))
            fig.update_layout(title="Noether's Theorem: Symmetry ↔ Conservation", xaxis_title="Symmetry", yaxis_title="Conserved Quantity")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "lotka_volterra":
            st.markdown("**Visualization: Lotka-Volterra Predator-Prey Model**")
            st.markdown("""
            **Equations:**
            \[
            \frac{dx}{dt} = \alpha x - \beta x y
            \]
            \[
            \frac{dy}{dt} = \delta x y - \gamma y
            \]
            where:
            - \( x \): Prey population
            - \( y \): Predator population
            - \( \alpha, \beta, \gamma, \delta \): Positive real parameters
            """)
            # Simple simulation
            from scipy.integrate import odeint

            def lotka_volterra(z, t, alpha, beta, gamma, delta):
                x, y = z
                dxdt = alpha * x - beta * x * y
                dydt = delta * x * y - gamma * y
                return [dxdt, dydt]
            
            alpha, beta, gamma, delta = 1.1, 0.4, 0.4, 0.1
            z0 = [10, 5]
            t = np.linspace(0, 15, 300)
            sol = odeint(lotka_volterra, z0, t, args=(alpha, beta, gamma, delta))
            x, y = sol.T
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=x, mode='lines', name='Prey Population'))
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Predator Population'))
            fig.update_layout(title="Lotka-Volterra Predator-Prey Model", xaxis_title="Time", yaxis_title="Population")
            st.plotly_chart(fig, use_container_width=True)
        
        elif v_type == "space_navigation":
            st.markdown("**Visualization: Deep Space Navigation Using Gravity Assists**")
            st.markdown("""
            **Example:** Voyager 1 used gravity assists to reach high speeds and navigate through the solar system.
            
            **Visualization:**
            """)
            st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Voyager_ecliptic.svg", caption="Voyager 1 Trajectory", use_column_width=True)
        
        elif v_type == "feynman_diagram":
            st.markdown("**Visualization: Feynman Diagram for Electron-Photon Interaction**")
            st.markdown("""
            **Description:** Feynman diagrams are pictorial representations of particle interactions in quantum field theory.
            """)
            st.image("https://upload.wikimedia.org/wikipedia/commons/0/0a/Feynman_diagram.svg", caption="Feynman Diagram", use_column_width=True)
        
        else:
            st.markdown("**Visualization for this concept is under development.**")
    
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
    
    st.header("Explore Mathematical Concepts and Their Physical Applications")
    st.markdown("""
    Select a mathematical concept from the dropdown menu below to explore its physical application through interactive visualizations and animations.
    """)
    
    # Dropdown to select a mathematical concept
    math_options = [
        "Newtonian Mechanics and Calculus",
        "Maxwell’s Equations and Electromagnetism",
        "General Relativity and Non-Euclidean Geometry",
        "Quantum Mechanics and Linear Algebra",
        "The Standard Model and Group Theory",
        "Fourier Analysis and Signal Processing",
        "Chaos Theory and Fractals",
        "Complex Numbers and Electrical Engineering",
        "Statistics and Machine Learning",
        "Euler’s Formula and Beauty",
        "Information Theory",
        "Prime Numbers and Cryptography",
        "Game Theory in Biology and Economics",
        "Statistical Mechanics and Thermodynamics",
        "Topology in Physics",
        "Neural Networks and Backpropagation",
        "Symmetry and the Universe (Noether's Theorem)",
        "Mathematical Biology",
        "Deep Space Navigation and Math",
        "Quantum Field Theory and Renormalization"
    ]
    
    selected_math = st.selectbox("Select a Mathematical Concept:", math_options)
    
    if selected_math:
        st.subheader(selected_math)
        # Find the concept in math_concepts
        concept = next((item for item in math_concepts if selected_math in item["title"]), None)
        
        if concept:
            st.markdown(f"**What?** {concept['what']}")
            st.markdown(f"**Unreasonable Effectiveness:** {concept['effectiveness']}")
            
            # Generate visualization based on type
            generate_visualization(concept["visualization_type"])
        else:
            st.markdown("**Visualization for this concept is under development.**")

# Quiz Section
elif selection == "Quiz":
    st.title("📝 Quiz: Test Your Understanding")
    
    # Define quiz questions
    quiz_questions = [
        {
            "question": "Which philosophical view posits that mathematical entities exist independently of human minds?",
            "options": ["Platonism", "Nominalism", "Constructivism", "Empiricism"],
            "answer": "Platonism"
        },
        {
            "question": "Which mathematical theory is essential for the formulation of quantum mechanics?",
            "options": ["Calculus", "Complex Numbers", "Topology", "Graph Theory"],
            "answer": "Complex Numbers"
        },
        {
            "question": "What does Non-Euclidean geometry primarily deal with?",
            "options": ["Flat surfaces", "Curved surfaces", "Two-dimensional spaces", "None of the above"],
            "answer": "Curved surfaces"
        },
        {
            "question": "Which mathematical concept is crucial for understanding the bending of light in General Relativity?",
            "options": ["Linear Algebra", "Differential Geometry", "Probability Theory", "Number Theory"],
            "answer": "Differential Geometry"
        },
        {
            "question": "Euler’s formula links five fundamental constants in mathematics. Which of the following is NOT one of them?",
            "options": ["e", "i", "π", "√2", "0"],
            "answer": "√2"
        },
        {
            "question": "Which mathematical method is used to predict weather patterns through the butterfly effect?",
            "options": ["Calculus", "Chaos Theory", "Group Theory", "Statistics"],
            "answer": "Chaos Theory"
        },
        {
            "question": "What does Noether's Theorem relate in physics?",
            "options": ["Energy and Mass", "Symmetry and Conservation Laws", "Force and Acceleration", "Entropy and Temperature"],
            "answer": "Symmetry and Conservation Laws"
        },
        {
            "question": "Which mathematical theory underpins modern encryption systems like RSA?",
            "options": ["Prime Numbers", "Fourier Analysis", "Neural Networks", "Topology"],
            "answer": "Prime Numbers"
        },
        {
            "question": "What mathematical framework is used to model animal behaviors and economic markets?",
            "options": ["Game Theory", "Statistics", "Linear Algebra", "Differential Equations"],
            "answer": "Game Theory"
        },
        {
            "question": "Which equation represents Euler’s Formula?",
            "options": ["e^{iπ} + 1 = 0", "F = ma", "E = mc²", "a² + b² = c²"],
            "answer": "e^{iπ} + 1 = 0"
        },
    ]
    
    # Initialize session state for quiz progress
    if 'quiz_progress' not in st.session_state:
        st.session_state.quiz_progress = {}
    
    # Iterate through questions
    for idx, q in enumerate(quiz_questions, 1):
        st.markdown(f"**Question {idx}:** {q['question']}")
        user_answer = st.radio(f"Select an option for Question {idx}:", q['options'], key=f"q{idx}")
        if st.button(f"Submit Q{idx}"):
            if user_answer == q['answer']:
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is {q['answer']}.")
        st.markdown("---")

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
    
    st.markdown("""
    **Educational Videos:**
    - [Khan Academy](https://www.khanacademy.org/)
    - [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
    - [Numberphile](https://www.youtube.com/user/numberphile)
    """)
    
    st.markdown("""
    **Interactive Simulations:**
    - [PhET Interactive Simulations](https://phet.colorado.edu/)
    - [Desmos Graphing Calculator](https://www.desmos.com/calculator)
    """)
    
    st.markdown("""
    **Research Papers:**
    - [Wigner's Original Paper](https://mathworld.wolfram.com/WignersUnreasonableEffectiveness.html)
    - [Noether's Theorem](https://en.wikipedia.org/wiki/Noether%27s_theorem)
    """)

# Footer
st.markdown("---")
st.markdown("""
    **Developed by:** [Your Name]
    **Contact:** your.email@example.com
    **GitHub:** [github.com/yourusername](https://github.com/yourusername)
""")

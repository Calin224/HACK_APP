from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional
import streamlit as st
import matplotlib.pyplot as plt
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import time
import numpy as np
from PIL import Image, ImageOps
import io
import copy
import random
import hashlib
import datetime

st.set_page_config(
    page_title="Criptografie Cuantica",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2 style="color: #667eea;">🚀 Quantum Tools</h2>
        <p style="opacity: 0.8;">Select your analysis mode</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.selectbox("Alege challenge-ul", [
        "🏠 Quantum Cryptography Overview",
        "🔍 Grover Attack Simulator",
        "🔢 Shor Implementation",
        "🚄 Performance & Optimization Lab",
        "🎮 Interactive Quantum Lab",
        "🧠 Grover vs Kyber (simulare educațională)",
        "🔥 Quantum Threat Dashboard"
    ])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">1,121</div>
        <div class="metric-label">IBM Condor Qubits</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Calculate simulated progress
    days_since_2024 = (datetime.datetime.now() - datetime.datetime(2024, 1, 1)).days
    progress = min(0.5 + (days_since_2024 * 0.001), 2.0)
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-number">{progress:.2f}%</div>
        <div class="metric-label">Quantum Progress</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">10⁶⁵x</div>
        <div class="metric-label">Grover Speedup</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">2³²</div>
        <div class="metric-label">AES-256 → AES-128</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: radial-gradient(ellipse at top, #0f0f23 0%, #1a1a2e 40%, #16213e 100%);
        min-height: 100vh;
        color: #ffffff;
    }
    
    .stApp {
        background: radial-gradient(ellipse at top, #0f0f23 0%, #1a1a2e 40%, #16213e 100%);
    }
    
    /* Enhanced Hero Section */
    .hero-container {
        position: relative;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 4rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 
            0 25px 50px rgba(102, 126, 234, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #ffffff, #f093fb, #f5f7fa, #ffeaa7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        position: relative;
        z-index: 2;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
        animation: textGlow 3s ease-in-out infinite alternate;
    }
    
    .hero-subtitle {
        font-size: 1.6rem;
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 2;
        font-family: 'Inter', sans-serif;
        margin-bottom: 2rem;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 2rem;
        position: relative;
        z-index: 2;
    }
    
    .hero-stat {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem 2rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(20px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover::before {
        transform: scaleX(1);
    }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-number {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', sans-serif;
        animation: numberPulse 2s ease-in-out infinite;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced Cards */
    .quantum-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        border-radius: 25px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(30px);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .quantum-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #ffeaa7);
        background-size: 200% 100%;
        animation: shimmer 3s ease-in-out infinite;
    }
    
    .quantum-card:hover {
        transform: translateY(-10px);
        box-shadow: 
            0 30px 60px rgba(102, 126, 234, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Algorithm Header */
    .algo-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Threat Level Cards */
    .threat-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: criticalPulse 2s ease-in-out infinite;
    }
    
    .threat-high {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(255, 167, 38, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .threat-medium {
        background: linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%);
        color: #2d3748;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(255, 235, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .threat-low {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .security-safe {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .attack-result {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(255, 167, 38, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .performance-metric {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
    }
    
    .advanced-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
    }
    
    .metric_box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
    }
    
    .box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        backdrop-filter: blur(20px);
    }
    
    .safe-box {
        background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Quantum Progress Bar */
    .quantum-progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 2rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .quantum-progress-bar {
        height: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #ffeaa7 75%, #00cec9 100%);
        background-size: 300% 100%;
        border-radius: 10px;
        animation: progressShimmer 3s ease-in-out infinite;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    .progress-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 12px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 6px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.6) !important;
        background: linear-gradient(45deg, #764ba2, #f093fb) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
    }
    
    /* Sidebar Enhancement */
    .css-1d391kg {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Form Controls */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stSlider > div > div > div {
        background: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Animations */
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes textGlow {
        from { text-shadow: 0 0 20px rgba(255, 255, 255, 0.3); }
        to { text-shadow: 0 0 40px rgba(255, 255, 255, 0.6); }
    }
    
    @keyframes numberPulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    @keyframes criticalPulse {
        0%, 100% { box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4); }
        50% { box-shadow: 0 20px 50px rgba(255, 107, 107, 0.7); }
    }
    
    @keyframes progressShimmer {
        0% { background-position: -300% 0; }
        100% { background-position: 300% 0; }
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px rgba(255, 107, 107, 0.3); }
        to { box-shadow: 0 0 30px rgba(255, 107, 107, 0.6); }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title { font-size: 2.5rem; }
        .hero-subtitle { font-size: 1.2rem; }
        .metric-number { font-size: 2.5rem; }
        .hero-stats { flex-direction: column; gap: 1rem; }
        .metrics-grid { grid-template-columns: 1fr; }
        .countdown-number { font-size: 3rem; }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 5px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #764ba2, #f093fb);
    }
    
    /* Loading Animation */
    .quantum-loader {
        display: inline-block;
        width: 40px;
        height: 40px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .quantum-countdown {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 20px 40px rgba(255, 107, 107, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    .countdown-number {
        font-size: 5rem;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class AttackResult:
    success: bool
    time_taken: int
    iterations: int
    method: str
    key_recovered: Optional[str] = None
    confidence: float = 0.0


@dataclass
class PerformanceMetrics:
    operation: str
    time_taken: int
    throughput: float
    memory_usage: float
    cpu_usage: float


def simulate_classical_search(n_bits):
    max_attempts = 2 ** (n_bits - 1)
    return max_attempts

def create_oracle(code: str):
    nr = len(code)

    oracle = QuantumCircuit(nr, name="oracle")

    for i, bit in enumerate(reversed(code)):
        if bit == '0':
            oracle.x(i)

    target = nr - 1
    controls = list(range(nr - 1))

    # apply MCZ = H + MCX + H
    oracle.h(target)
    oracle.mcx(controls, target)
    oracle.h(target)

    for i, bit in enumerate(reversed(code)):
        if bit == '0':
            oracle.x(i)

    return oracle

def create_diffusion(code):  # H - X - MCZ - X - H
    nr = len(code)

    diff = QuantumCircuit(nr, name="diffusion")

    all_qubits = range(nr)
    target = nr - 1
    controls = list(range(nr - 1))

    diff.h(all_qubits)
    diff.x(all_qubits)

    diff.h(target)
    diff.mcx(controls, target)
    diff.h(target)

    diff.x(all_qubits)
    diff.h(all_qubits)

    return diff

def run_grover_algo(code):  # H - oracle - diff
    qubits = len(code)

    grover = QuantumCircuit(qubits)
    grover.h(range(qubits))

    num_iterations = math.ceil(math.pi / 4 * math.sqrt(2 ** len(code)))

    for _ in range(num_iterations):
        grover.append(create_oracle(code), list(range(len(code))))
        grover.append(create_diffusion(code), list(range(len(code))))

    grover.measure_all()

    backend = AerSimulator()
    transpiled_grover = transpile(grover, backend)
    job = backend.run(transpiled_grover)
    result = job.result()
    counts = result.get_counts()

    return grover, counts, num_iterations

def c_amod15(a, power):
    U = QuantumCircuit(4, name=f"{a}^{power} mod 15")

    for iter in range(power):
        U.swap(0, 1)
        U.swap(1, 2)
        U.swap(2, 3)

        for q in range(4):
            U.x(q)

    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U

def qft_dagger(n):
    qc = QuantumCircuit(n)
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)

    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / float(2 ** (j - m)), m, j)
        qc.h(j)

    qc.name = "dagger"
    return qc

def run_shor_algo(n_count=8, a=7):
    qc = QuantumCircuit(n_count + 4, n_count)

    for q in range(n_count):
        qc.h(q)

    qc.x(n_count)

    for q in range(n_count):
        qc.append(c_amod15(a, 2 ** q), [q] + [i + n_count for i in range(4)])

    qc.append(qft_dagger(n_count), range(n_count))
    qc.measure(range(n_count), range(n_count))

    backend = AerSimulator()
    transpiled_res = transpile(qc, backend)
    job = backend.run(transpiled_res, shots=1024)
    result = job.result()
    counts = result.get_counts()

    return qc, counts

def run_grover_with_progress(code, callback=None):
    qubits = len(code)
    grover = QuantumCircuit(qubits)
    grover.h(range(qubits))

    num_iterations = math.ceil(math.pi / 4 * math.sqrt(2 ** len(code)))
    iteration_data = []

    target_int = int(code, 2)

    for i in range(num_iterations):
        grover.append(create_oracle(code), list(range(len(code))))
        grover.append(create_diffusion(code), list(range(len(code))))

        angle = (2 * i + 1) * math.asin(1 / math.sqrt(2 ** qubits))
        prob = math.sin(angle) ** 2

        iteration_data.append({
            'iteration': i + 1,
            'target_probability': prob,
            'amplification': prob / (1/2 ** qubits),
            'circuit_depth': grover.depth(), 
        })

        if callback:
            callback(i + 1, prob, num_iterations)

    grover.measure_all()

    backend = AerSimulator()
    transpiled_grover = transpile(grover, backend)
    job = backend.run(transpiled_grover, shots=2048)
    result = job.result()
    counts = result.get_counts()

    return grover, counts, num_iterations, iteration_data

def create_live_grover_plot(iteration_data):
    df = pd.DataFrame(iteration_data)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Target Probability Evolution', 'Amplitude Amplification', 
                       'Circuit Depth Growth', 'Quantum Advantage'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    fig.add_trace(
        go.Scatter(x=df['iteration'], y=df['target_probability'],
                  mode='lines+markers', name='Target Prob',
                  line=dict(color='#00ff87', width=3)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['iteration'], y=df['amplification'],
                  mode='lines+markers', name='Amplification',
                  line=dict(color='#ff6b6b', width=3)),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=df['iteration'], y=df['circuit_depth'],
                  mode='lines+markers', name='Depth',
                  line=dict(color='#4a9eff', width=3)),
        row=2, col=1
    )

    n_qubits = len(iteration_data)
    classical_ops = [2**(i//2) for i in df['iteration']]
    quantum_ops = df['iteration'].tolist()

    fig.add_trace(
        go.Scatter(x=df['iteration'], y=classical_ops,
                  mode='lines', name='Classical',
                  line=dict(color='#ff8c00', width=2, dash='dash')),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(x=df['iteration'], y=quantum_ops,
                  mode='lines+markers', name='Quantum',
                  line=dict(color='#667eea', width=3)),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        title_text="🔍 Real-Time Grover Algorithm Analysis",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    return fig

def analyze_shor_results(counts, n_count, a_val, N=15):
    analysis_results = {
        'measurements': [],
        'period_candidates': [],
        'factorization_attempts': [],
        'success_probability': 0.0, 
        'quantum_advantage': 0.0
    }

    total_shots = sum(counts.values())

    for binary_str, count in counts.items():
        decimal_val = int(binary_str, 2)
        probability = count / total_shots  

        if decimal_val != 0:
            estimated_period = 2 ** n_count / decimal_val 
            rounded_period = round(estimated_period)

            period_accuracy = 1.0 - abs(estimated_period - rounded_period) / max(rounded_period, 1)

            measurement_data = { 
                'binary': binary_str,
                'decimal': decimal_val,
                'count': count,
                'probability': probability,
                'estimated_period': estimated_period,
                'rounded_period': rounded_period,
                'period_accuracy': period_accuracy,
                'quantum_phase': decimal_val / (2 ** n_count)
            }

            analysis_results['measurements'].append(measurement_data)

            if rounded_period > 0 and rounded_period <= 20 and period_accuracy > 0.7:
                factors = attempt_factorization(a_val, N, rounded_period)  # FIXAT: parametrii
                if factors:
                    analysis_results['factorization_attempts'].append({
                        'period': rounded_period,
                        'factors': factors,
                        'probability': probability,
                        'success': factors[0] * factors[1] == N
                    })

    analysis_results['measurements'].sort(key=lambda x: x['probability'], reverse=True)

    successful_attempts = [a for a in analysis_results['factorization_attempts'] if a['success']]
    if successful_attempts:
        analysis_results['success_probability'] = sum(a['probability'] for a in successful_attempts)

    classical_time = math.sqrt(N)
    quantum_time = (math.log(N) ** 3) * (n_count ** 2)
    analysis_results['quantum_advantage'] = classical_time / quantum_time

    return analysis_results

def attempt_factorization(a, N, r):
    if r <= 0 or r % 2 != 0: 
        return None
    
    x = pow(a, r // 2, N)
    if x == 1 or x == N - 1: 
        return None
    
    factor1 = math.gcd(x - 1, N)
    factor2 = math.gcd(x + 1, N)

    if factor1 > 1 and factor1 < N:
        return (factor1, N // factor1)
    if factor2 > 1 and factor2 < N:
        return (factor2, N // factor2)
    
    return None

def create_quantum_state_visualizer(circuit):
    try:
        viz_circuit = circuit.copy()
        
        viz_circuit.data = [instruction for instruction in viz_circuit.data 
                           if instruction.operation.name not in ['measure', 'barrier']]
        
        viz_circuit.cregs = []
        
        n_qubits = viz_circuit.num_qubits
        if n_qubits > 10:  
            st.warning(f"Circuit has {n_qubits} qubits. Limiting visualization to first 8 qubits for performance.")
            truncated_circuit = QuantumCircuit(8)
            for instruction in viz_circuit.data:
                if all(qubit.index < 8 for qubit in instruction.qubits):
                    truncated_circuit.append(instruction.operation, 
                                           [truncated_circuit.qubits[q.index] for q in instruction.qubits])
            viz_circuit = truncated_circuit
            n_qubits = 8
        
        psi = Statevector.from_instruction(viz_circuit)
        amplitudes = psi.data
        probabilities = psi.probabilities()
        
        states = [format(i, f'0{n_qubits}b') for i in range(len(amplitudes))]
        
        significant_states = [(i, states[i], probabilities[i], amplitudes[i]) 
                            for i in range(len(states)) 
                            if probabilities[i] > 1e-6]
        
        if not significant_states:
            return None
            
        significant_states.sort(key=lambda x: x[2], reverse=True)
        
        if len(significant_states) > 20:
            significant_states = significant_states[:20]
            
        indices, state_labels, probs, amps = zip(*significant_states)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=state_labels,
            y=probs,
            name='Probabilities',
            marker_color='rgba(0, 255, 135, 0.7)',
            text=[f'{p:.4f}' for p in probs],
            textposition='auto',
            hovertemplate='State: %{x}<br>Probability: %{y:.6f}<extra></extra>'
        ))
        
        real_parts = [amp.real for amp in amps]
        imag_parts = [amp.imag for amp in amps]
        
        fig.add_trace(go.Scatter(
            x=state_labels,
            y=real_parts,
            mode='markers',
            name='Real Part',
            marker=dict(color='#ff6b6b', size=8, symbol='circle'),
            yaxis='y2',
            hovertemplate='State: %{x}<br>Real: %{y:.6f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=state_labels,
            y=imag_parts,
            mode='markers',
            name='Imaginary Part', 
            marker=dict(color='#4a9eff', size=8, symbol='diamond'),
            yaxis='y2',
            hovertemplate='State: %{x}<br>Imaginary: %{y:.6f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f"🌊 Quantum State Visualization ({len(significant_states)} states)",
                'font': {'size': 18, 'color': 'white'}
            },
            xaxis_title="Quantum States",
            yaxis_title="Probability",
            yaxis2=dict(
                title="Amplitude", 
                overlaying='y', 
                side='right',
                range=[min(min(real_parts), min(imag_parts)) * 1.1, 
                       max(max(real_parts), max(imag_parts)) * 1.1]
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        print(f"Error in quantum state visualization: {str(e)}")
        return None

def create_amplitude_phase_plot(circuit):
    try:
        viz_circuit = circuit.copy()
        viz_circuit.data = [instruction for instruction in viz_circuit.data 
                           if instruction.operation.name not in ['measure', 'barrier']]
        viz_circuit.cregs = []
        
        n_qubits = viz_circuit.num_qubits
        if n_qubits > 8:
            return None
            
        psi = Statevector.from_instruction(viz_circuit)
        amplitudes = psi.data
        
        magnitudes = np.abs(amplitudes)
        phases = np.angle(amplitudes)
        
        states = [format(i, f'0{n_qubits}b') for i in range(len(amplitudes))]
        
        significant = [(i, states[i], magnitudes[i], phases[i]) 
                      for i in range(len(states)) 
                      if magnitudes[i] > 1e-6]
        
        if not significant:
            return None
            
        significant.sort(key=lambda x: x[2], reverse=True)
        if len(significant) > 16:
            significant = significant[:16]
            
        indices, state_labels, mags, phase_vals = zip(*significant)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Amplitude Magnitudes', 'Phase Angles'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=state_labels, y=mags, name='|Amplitude|',
                  marker_color='rgba(102, 126, 234, 0.8)'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=state_labels, y=phase_vals, mode='markers+lines',
                      name='Phase', marker=dict(size=10, color='#f093fb')),
            row=1, col=2
        )
        
        fig.update_layout(
            title="🎯 Quantum Amplitude & Phase Analysis",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=400
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in amplitude/phase visualization: {str(e)}")
        return None

def enhanced_circuit_analysis(circuit, target_state=None):
    analysis_results = {}
    
    try:
        analysis_results['num_qubits'] = circuit.num_qubits
        analysis_results['depth'] = circuit.depth()
        analysis_results['gate_count'] = len(circuit.data)
        
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
        analysis_results['gate_composition'] = gate_counts
        
        viz_circuit = circuit.copy()
        viz_circuit.data = [instruction for instruction in viz_circuit.data 
                           if instruction.operation.name not in ['measure', 'barrier']]
        viz_circuit.cregs = []
        
        if viz_circuit.num_qubits <= 10:
            psi = Statevector.from_instruction(viz_circuit)
            analysis_results['statevector'] = psi
            analysis_results['probabilities'] = psi.probabilities()
            
            probs = psi.probabilities()
            nonzero_probs = probs[probs > 1e-12]
            entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
            analysis_results['entropy'] = entropy
            
            if target_state:
                target_int = int(target_state, 2)
                target_prob = probs[target_int] if target_int < len(probs) else 0
                analysis_results['target_probability'] = target_prob
                analysis_results['target_fidelity'] = np.sqrt(target_prob)
                
    except Exception as e:
        analysis_results['error'] = str(e)
    
    return analysis_results
    
def create_performance_comparison(classical_time, quantum_time, problem_size):
    fig = go.Figure()
    
    methods = ['Classical Brute Force', 'Quantum Algorithm']
    times = [classical_time, quantum_time]
    colors = ['#ff6b6b', '#00ff87']
    
    fig.add_trace(go.Bar(
        x=methods,
        y=[math.log10(t) for t in times],
        marker_color=colors,
        text=[f'{t:.2e}s' if t > 1e6 else f'{t:.2f}s' for t in times],
        textposition='auto'
    ))
    
    speedup = classical_time / quantum_time if quantum_time > 0 else float('inf')
    
    fig.add_annotation(
        x=0.5, y=max(math.log10(t) for t in times) * 0.8,
        text=f"🚀 Quantum Speedup: {speedup:.2e}x",
        showarrow=False,
        font=dict(size=16, color='#f093fb'),
        bgcolor='rgba(240, 147, 251, 0.2)',
        bordercolor='#f093fb',
        borderwidth=2
    )
    
    fig.update_layout(
        title=f"⚡ Performance Comparison (Problem Size: {problem_size})",
        yaxis_title="Time (log10 seconds)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400
    )
    
    return fig
        
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🔮 QuantumCrypt</h1>
    <p class="hero-subtitle">Next-Generation Quantum Cryptography Analysis Platform</p>
    <div class="hero-stats">
        <div class="hero-stat">
            <strong>Quantum Ready</strong><br>
            <small>Advanced Algorithms</small>
        </div>
        <div class="hero-stat">
            <strong>Real-Time</strong><br>
            <small>Live Analysis</small>
        </div>
        <div class="hero-stat">
            <strong>Secure</strong><br>
            <small>Post-Quantum</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


if page == "🏠 Quantum Cryptography Overview":
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("""
        <div class="quantum-card">
            <h2>🌟 Welcome to the Quantum Era</h2>
            <p>Experience cutting-edge quantum cryptanalysis tools. This suite demonstrates real quantum algorithms that will reshape cybersecurity.</p>
            <h4>🎯 What You'll Discover:</h4>
            <ol>
                <li><strong>Grover's Algorithm:</strong> Quadratic speedup for search</li>
                <li><strong>Shor's Algorithm:</strong> Exponential advantage for factoring</li>
                <li><strong>Post-Quantum Crypto:</strong> Future-proof security</li>
                <li><strong>Real-Time Analysis:</strong> Live threat assessment</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="advanced-box">
            <h2>⚡ Quantum Advantage Analysis</h2>
            <p>Real-time comparison of classical vs quantum computational complexity across different cryptographic primitives.</p>
        </div>
        """, unsafe_allow_html=True)

        fig = go.Figure()
        n_values = np.arange(1, 21)

        classical_search = 2 ** n_values
        quantum_search = np.sqrt(2 ** n_values)

        classical_factoring = np.exp(
            1.9 * ((n_values * np.log(2)) ** (1 / 3)) * (np.log(n_values * np.log(2)) ** (2 / 3)))
        quantum_factoring = n_values ** 3

        fig.add_trace(go.Scatter(x=n_values, y=classical_search, name="Classical Search O(2ⁿ)",
                                 line=dict(color='#e53e3e', width=3)))
        fig.add_trace(
            go.Scatter(x=n_values, y=quantum_search, name="Quantum Search O(√2ⁿ)", line=dict(color='#667eea', width=3)))
        fig.add_trace(go.Scatter(x=n_values, y=classical_factoring, name="Classical Factoring",
                                 line=dict(color='#ff8c00', width=3, dash='dot')))
        fig.add_trace(go.Scatter(x=n_values, y=quantum_factoring, name="Quantum Factoring O(n³)",
                                 line=dict(color='#32cd32', width=3, dash='dash')))

        fig.update_layout(
            title={
                'text': "⚡ Quantum vs Classical Complexity",
                'font': {'size': 20, 'color': 'white'}
            },
            xaxis_title="Problem Size (bits)",
            yaxis_title="Operations",
            yaxis_type="log",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Grover Attack Simulator":
    st.markdown("<h2 class='algo-header'>🔍 Grover Attack Simulator</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🎮 Attack Configuration")

        attack_mode = st.selectbox("Attack Mode", [
            "🎯 Targeted Search",
            "🔄 Multi-Target Attack"
        ])

        qubits = st.slider("Number of qubits", 2, 12, 6)

        if attack_mode == "🎯 Targeted Search":
            secret_code = st.text_input("Codul secret care trebuie gasit: ", value="0" * qubits, max_chars=qubits)

            if not all(c in '01' for c in secret_code):
                st.error("Secret code can only contain 0 and 1!")
            elif len(secret_code) != qubits:
                st.error(f"Secret code must be exactly {qubits} bits long!")
            else:
                targets = [secret_code]
        else:
            num_targets = st.slider("Number of targets", 2, 5, 3)

            if st.button("🎯 Generate Multi-targets"):
                st.session_state.multi_targets = []
                for i in range(num_targets):
                    target = ''.join(random.choice('01') for _ in range(qubits))
                    st.session_state.multi_targets.append(target)
                st.success(f"Generated {num_targets} multi-targets!")

            if 'multi_targets' in st.session_state:
                st.markdown("**Target Set:**")
                for i, target in enumerate(st.session_state.multi_targets):
                    st.code(f"Target {i + 1}: {target}")
                    targets = st.session_state.multi_targets
                    secret_code = targets[0]
            else:
                targets = []
                secret_code = '0' * qubits

        if st.button("🚀 Launch Quantum Attack!", type="primary"):

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(iteration, probability, total):
                progress = iteration / total
                progress_bar.progress(progress)
                status_text.text(f"🔄 Iteration {iteration}/{total} - Target Probability: {probability:.4f}")

            with st.spinner("Quantum computation in progress..."):
                start_time = time.time()

                if attack_mode == "🎯 Targeted Search":
                    st.markdown("### 🎯 Executing Enhanced Targeted Search...")
                    
                    circuit, counts, iterations, iteration_data = run_grover_with_progress(
                        secret_code, progress_callback
                    )
                    
                    execution_time = time.time() - start_time
                    
                    progress_bar.progress(1.0)
                    status_text.success("✅ Quantum attack completed!")
                    
                    cel_mai_cautat = max(counts.items(), key=lambda x: x[1])
                    success = cel_mai_cautat[0] == secret_code
                    succes_prob = cel_mai_cautat[1] / sum(counts.values())

                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("🎯 Found Value", cel_mai_cautat[0])
                    with col_b:
                        st.metric("🔄 Quantum Iterations", iterations)
                    with col_c:
                        st.metric("📊 Success Probability", f"{succes_prob:.3f}")
                    with col_d:
                        classical_attempt = 2 ** (qubits - 1)
                        speedup = classical_attempt // iterations
                        st.metric("⚡ Quantum Speedup", f"{speedup}x")

                    st.markdown("### 📈 Real-Time Algorithm Analysis")
                    st.plotly_chart(create_live_grover_plot(iteration_data), use_container_width=True)
                    
                    classical_time = 2 ** (qubits - 1) / 1e9 
                    quantum_time = execution_time
                    
                    st.plotly_chart(
                        create_performance_comparison(classical_time, quantum_time, qubits),
                        use_container_width=True
                    )

                    st.markdown("### 🌊 Enhanced Quantum State Analysis")
                    
                    analysis = enhanced_circuit_analysis(circuit, secret_code)
                    
                    col_stats1, col_stats2, col_stats3 = st.columns(3)
                    with col_stats1:
                        st.metric("🔧 Circuit Depth", analysis.get('depth', 'N/A'))
                    with col_stats2:
                        st.metric("🎛️ Total Gates", analysis.get('gate_count', 'N/A'))
                    with col_stats3:
                        if 'entropy' in analysis:
                            st.metric("🌀 Quantum Entropy", f"{analysis['entropy']:.3f}")
                    
                    if 'gate_composition' in analysis:
                        st.markdown("#### 🔧 Gate Composition Analysis")
                        gate_df = pd.DataFrame(list(analysis['gate_composition'].items()), 
                                              columns=['Gate Type', 'Count'])
                        gate_df = gate_df.sort_values('Count', ascending=False)
                        
                        col_gate1, col_gate2 = st.columns([1, 1])
                        
                        with col_gate1:
                            fig_gates = go.Figure(data=[
                                go.Bar(x=gate_df['Gate Type'], y=gate_df['Count'],
                                       marker_color='rgba(102, 126, 234, 0.8)',
                                       text=gate_df['Count'],
                                       textposition='auto')
                            ])
                            fig_gates.update_layout(
                                title="Gate Usage Distribution",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=300
                            )
                            st.plotly_chart(fig_gates, use_container_width=True)
                        
                        with col_gate2:
                            total_gates = gate_df['Count'].sum()
                            gate_df['Percentage'] = (gate_df['Count'] / total_gates * 100).round(1)
                            
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=gate_df['Gate Type'], 
                                values=gate_df['Count'],
                                hole=0.4,
                                marker_colors=['#667eea', '#764ba2', '#f093fb', '#ffeaa7', '#00cec9']
                            )])
                            fig_pie.update_layout(
                                title="Gate Type Distribution",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=300,
                                showlegend=True
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    state_viz = create_quantum_state_visualizer(circuit)
                    if state_viz:
                        st.plotly_chart(state_viz, use_container_width=True)
                        
                        if 'target_probability' in analysis:
                            target_prob = analysis['target_probability']
                            target_fidelity = analysis['target_fidelity']
                            
                            col_target1, col_target2 = st.columns([1, 1])
                            
                            with col_target1:
                                st.markdown(f"""
                                <div class="quantum-card">
                                    <h4>🎯 Target State Analysis</h4>
                                    <p><strong>Target State:</strong> |{secret_code}⟩</p>
                                    <p><strong>Probability:</strong> {target_prob:.6f}</p>
                                    <p><strong>Fidelity:</strong> {target_fidelity:.6f}</p>
                                    <p><strong>Success Rate:</strong> {target_prob * 100:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_target2:
                                fig_gauge = go.Figure(go.Indicator(
                                    mode = "gauge+number+delta",
                                    value = target_prob * 100,
                                    domain = {'x': [0, 1], 'y': [0, 1]},
                                    title = {'text': "Success Probability (%)"},
                                    delta = {'reference': 50},
                                    gauge = {
                                        'axis': {'range': [None, 100]},
                                        'bar': {'color': "#667eea"},
                                        'steps': [
                                            {'range': [0, 25], 'color': "#ff6b6b"},
                                            {'range': [25, 75], 'color': "#ffa726"},
                                            {'range': [75, 100], 'color': "#4caf50"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "white", 'width': 4},
                                            'thickness': 0.75,
                                            'value': 90
                                        }
                                    }
                                ))
                                fig_gauge.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': 'white'},
                                    height=300
                                )
                                st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        amp_phase_viz = create_amplitude_phase_plot(circuit)
                        if amp_phase_viz:
                            st.markdown("#### 📊 Detailed Amplitude & Phase Analysis")
                            st.plotly_chart(amp_phase_viz, use_container_width=True)
                    
                    else:
                        st.markdown("""
                        <div class="quantum-card">
                            <h4>⚠️ Circuit Complexity Notice</h4>
                            <p>Circuit is too complex for detailed state visualization. Here's comprehensive analysis:</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>📊 Circuit Metrics</h4>
                                <p><strong>Qubits:</strong> {analysis.get('num_qubits', 'N/A')}</p>
                                <p><strong>Depth:</strong> {analysis.get('depth', 'N/A')}</p>
                                <p><strong>Gates:</strong> {analysis.get('gate_count', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_info2:
                            estimated_time = analysis.get('gate_count', 0) * 0.1
                            memory_estimate = (2 ** min(qubits, 20)) * 8 / (1024**2)  # MB
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>⚡ Resource Estimates</h4>
                                <p><strong>Runtime:</strong> {estimated_time:.2f} μs</p>
                                <p><strong>Memory:</strong> {memory_estimate:.1f} MB</p>
                                <p><strong>Complexity:</strong> O(√2^{qubits})</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_info3:
                            classical_complexity = 2 ** qubits
                            quantum_complexity = math.sqrt(2 ** qubits)
                            advantage = classical_complexity / quantum_complexity
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>🚀 Quantum Advantage</h4>
                                <p><strong>Classical:</strong> 2^{qubits}</p>
                                <p><strong>Quantum:</strong> √2^{qubits}</p>
                                <p><strong>Speedup:</strong> {advantage:.0f}x</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📈 Enhanced Measurement Analysis")
                        
                        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
                        top_results = dict(list(sorted_counts.items())[:12]) 
                        
                        total_shots = sum(counts.values())
                        target_count = counts.get(secret_code, 0)
                        target_rank = list(sorted_counts.keys()).index(secret_code) + 1 if secret_code in sorted_counts else len(sorted_counts)
                        
                        col_results1, col_results2 = st.columns([2, 1])
                        
                        with col_results1:
                            fig_results = go.Figure(data=[
                                go.Bar(
                                    x=list(top_results.keys()), 
                                    y=list(top_results.values()),
                                    marker_color=[
                                        '#00ff87' if k == secret_code else 
                                        '#667eea' if i < 3 else 
                                        '#764ba2' for i, k in enumerate(top_results.keys())
                                    ],
                                    text=[f'{v}/{total_shots}<br>({v/total_shots*100:.1f}%)' for v in top_results.values()],
                                    textposition='auto',
                                    hovertemplate='State: %{x}<br>Count: %{y}<br>Probability: %{text}<extra></extra>'
                                )
                            ])
                            
                            fig_results.update_layout(
                                title="Top Measurement Outcomes Analysis",
                                xaxis_title="Quantum States",
                                yaxis_title="Measurement Count",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=400
                            )
                            
                            if secret_code in top_results:
                                fig_results.add_annotation(
                                    x=secret_code,
                                    y=target_count,
                                    text="🎯 TARGET",
                                    showarrow=True,
                                    arrowhead=2,
                                    arrowcolor="#00ff87",
                                    font=dict(color="#00ff87", size=12)
                                )
                            
                            st.plotly_chart(fig_results, use_container_width=True)
                        
                        with col_results2:
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>📊 Results Statistics</h4>
                                <p><strong>Target Found:</strong> {'✅ Yes' if target_count > 0 else '❌ No'}</p>
                                <p><strong>Target Rank:</strong> #{target_rank}</p>
                                <p><strong>Target Count:</strong> {target_count}/{total_shots}</p>
                                <p><strong>Success Rate:</strong> {target_count/total_shots*100:.2f}%</p>
                                <p><strong>Unique States:</strong> {len(counts)}</p>
                                <p><strong>Max Count:</strong> {max(counts.values())}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            probs = list(counts.values())
                            prob_array = np.array(probs) / total_shots
                            entropy = -np.sum(prob_array * np.log2(prob_array + 1e-12))
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>🌀 Distribution Analysis</h4>
                                <p><strong>Entropy:</strong> {entropy:.3f} bits</p>
                                <p><strong>Uniformity:</strong> {entropy/qubits*100:.1f}%</p>
                                <p><strong>Concentration:</strong> {max(probs)/total_shots*100:.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with st.expander("🔬 Advanced Quantum Circuit Analysis"):
                        if 'error' in analysis:
                            st.error(f"Analysis Error: {analysis['error']}")
                        
                        col_debug1, col_debug2 = st.columns([1, 1])
                        
                        with col_debug1:
                            st.markdown("#### 🧪 Circuit Debug Information")
                            st.code(f"""
                            Circuit Specifications:
                            - Qubits: {circuit.num_qubits}
                            - Classical bits: {circuit.num_clbits}  
                            - Depth: {circuit.depth()}
                            - Size: {circuit.size()}
                            - Operations: {len(circuit.data)}

                            Grover-Specific Metrics:
                            - Target: |{secret_code}⟩
                            - Search Space: 2^{qubits} = {2**qubits} states
                            - Optimal Iterations: {iterations}
                            - Theoretical Success: ~100%
                            - Measured Success: {succes_prob*100:.1f}%

                            Gate Breakdown:
                            {chr(10).join([f"- {gate}: {count}" for gate, count in analysis.get('gate_composition', {}).items()])}
                            """)
                        
                        with col_debug2:
                            st.markdown("#### 📐 Quantum Algorithm Theory")
                            
                            theoretical_prob = math.sin((2*iterations + 1) * math.asin(1/math.sqrt(2**qubits)))**2
                            
                            theory_data = pd.DataFrame({
                                'Metric': ['Success Probability', 'Iterations', 'Amplification', 'Fidelity'],
                                'Theoretical': [f'{theoretical_prob:.4f}', f'{iterations}', f'{theoretical_prob * 2**qubits:.1f}x', '1.000'],
                                'Measured': [f'{succes_prob:.4f}', f'{iterations}', f'{succes_prob * 2**qubits:.1f}x', f'{math.sqrt(succes_prob):.3f}']
                            })
                            
                            st.dataframe(theory_data, use_container_width=True)
                            
                            if circuit.num_qubits <= 6:
                                st.markdown("#### 🔧 Circuit Visualization")
                                try:
                                    fig_circuit = circuit.draw(output='mpl', fold=-1)
                                    st.pyplot(fig_circuit, use_container_width=True)
                                except:
                                    st.code(str(circuit.draw(fold=-1)))
                            else:
                                st.info(f"Circuit too large ({circuit.num_qubits} qubits) for diagram display")

                    if success:
                        st.markdown('''
                        <div class="security-safe">
                            <h4>🎉 Quantum Attack Successful!</h4>
                            <p>Grover's algorithm successfully found the target state with maximum probability! This demonstrates the power of quantum amplitude amplification.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('''
                        <div class="attack-result">
                            <h4>⚠️ Partial Success - Quantum Nature</h4>
                            <p>Algorithm found a high-probability solution, but not the exact target. This demonstrates the probabilistic nature of quantum algorithms and the importance of multiple runs.</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                else:
                    st.markdown("### 🔄 Executing Multi-target Attack...")

                    start_time = time.time()

                    def create_multi_oracle(targets_list, qubits):
                        oracle = QuantumCircuit(qubits, name="multi-oracle")

                        for target in targets_list:
                            for i, bit in enumerate(reversed(target)):
                                if bit == '0':
                                    oracle.x(i)

                            if qubits > 1:
                                oracle.h(qubits - 1)
                                if qubits > 2:
                                    oracle.mcx(list(range(qubits - 1)), qubits - 1)
                                else:
                                    oracle.cx(0, 1)
                                oracle.h(qubits - 1)

                            for i, bit in enumerate(reversed(target)):
                                if bit == '0':
                                    oracle.x(i)

                        return oracle

                    def run_multi_grover(targets_list, qubits):
                        grover = QuantumCircuit(qubits)
                        grover.h(range(qubits))

                        num_iterations = max(1, math.ceil(math.pi / 4 * math.sqrt(2 ** qubits / len(targets_list))))

                        for _ in range(num_iterations):
                            grover.append(create_multi_oracle(targets_list, qubits), list(range(qubits)))
                            grover.append(create_diffusion("0" * qubits), list(range(qubits)))

                        grover.measure_all()

                        backend = AerSimulator()
                        transpiled_grover = transpile(grover, backend)
                        job = backend.run(transpiled_grover, shots=2048)
                        result = job.result()
                        counts = result.get_counts()

                        return grover, counts, num_iterations

                    circuit, counts, iterations = run_multi_grover(targets, qubits)
                    attack_time = time.time() - start_time

                    target_results = {}
                    total_shots = sum(counts.values())

                    for target in targets:
                        if target in counts:
                            prob = counts[target] / total_shots
                            target_results[target] = prob
                        else:
                            target_results[target] = 0.0

                    most_prob = max(counts.items(), key=lambda x: x[1])
                    found_targets = [target for target in targets if target in counts.keys()]

                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("🎯 Targets Found", f"{len(found_targets)}/{len(targets)}")
                    with col_b:
                        st.metric("🔄 Quantum Iterations", iterations)
                    with col_c:
                        st.metric("📊 Best Probability", f"{most_prob[1] / total_shots:.3f}")
                    with col_d:
                        classical_ops = 2 ** (qubits - 1)
                        speedup = classical_ops // iterations
                        st.metric("⚡ Quantum Speedup", f"{speedup}x")

                    st.markdown("### 🎯 Individual Target Results")
                    for target in targets:
                        prob = target_results[target]
                        found = target in counts
                        status = "✅ Found" if found else "❌ Not Found"

                        st.markdown(f"""
                        <div class="{'security-safe' if found else 'attack-result'}">
                            <h4>Target: {target} - {status}</h4>
                            <p><strong>Probability:</strong> {prob:.3f}</p>
                            <p><strong>Measurements:</strong> {counts.get(target, 0)}/{total_shots}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### 🎯 Enhanced Multi-Target Analysis")
                    
                    multi_analysis = enhanced_circuit_analysis(circuit, targets[0])
                    
                    col_multi1, col_multi2, col_multi3, col_multi4 = st.columns(4)
                    
                    with col_multi1:
                        st.metric("🎯 Total Targets", len(targets))
                    with col_multi2:
                        st.metric("✅ Found Targets", len(found_targets))
                    with col_multi3:
                        efficiency = len(found_targets) / len(targets) * 100
                        st.metric("📊 Efficiency", f"{efficiency:.1f}%")
                    with col_multi4:
                        avg_prob = sum(target_results.values()) / len(targets)
                        st.metric("⚡ Avg Probability", f"{avg_prob:.3f}")
                    
                    st.markdown("#### 🎯 Target Performance Comparison")
                    
                    col_comp1, col_comp2 = st.columns([2, 1])
                    
                    with col_comp1:
                        target_names = [f"Target_{i+1}\n{target}" for i, target in enumerate(targets)]
                        probabilities = [target_results[target] for target in targets]
                        found_status = [target in counts for target in targets]
                        measurement_counts = [counts.get(target, 0) for target in targets]
                        
                        fig_targets = go.Figure()
                        
                        fig_targets.add_trace(go.Bar(
                            name='Probability',
                            x=target_names,
                            y=probabilities,
                            marker_color=['#00ff87' if found else '#ff6b6b' for found in found_status],
                            text=[f'{p:.3f}' for p in probabilities],
                            textposition='auto',
                            yaxis='y'
                        ))
                        
                        fig_targets.add_trace(go.Scatter(
                            name='Measurements',
                            x=target_names,
                            y=measurement_counts,
                            mode='markers+lines',
                            marker=dict(size=10, color='#f093fb'),
                            line=dict(color='#f093fb', width=2),
                            yaxis='y2'
                        ))
                        
                        fig_targets.update_layout(
                            title="Multi-Target Success Analysis",
                            xaxis_title="Targets",
                            yaxis_title="Probability",
                            yaxis2=dict(title="Measurement Count", overlaying='y', side='right'),
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=400,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_targets, use_container_width=True)
                    
                    with col_comp2:
                        success_data = ['Found', 'Not Found']
                        success_values = [len(found_targets), len(targets) - len(found_targets)]
                        
                        fig_success = go.Figure(data=[go.Pie(
                            labels=success_data,
                            values=success_values,
                            hole=0.5,
                            marker_colors=['#4caf50', '#ff6b6b']
                        )])
                        
                        fig_success.update_layout(
                            title="Success Rate Distribution",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300,
                            annotations=[dict(text=f'{efficiency:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
                        )
                        
                        st.plotly_chart(fig_success, use_container_width=True)
                    
                    st.markdown("#### 🌊 Multi-Target Quantum State Analysis")
                    
                    multi_state_viz = create_quantum_state_visualizer(circuit)
                    if multi_state_viz:
                        st.plotly_chart(multi_state_viz, use_container_width=True)
                        
                        st.markdown("""
                        <div class="quantum-card">
                            <h4>🎯 Multi-Target State Analysis</h4>
                            <p>The visualization above shows the quantum superposition where multiple target states are simultaneously amplified. Green markers indicate successfully found targets.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    else:
                        col_fallback1, col_fallback2 = st.columns([1, 1])
                        
                        with col_fallback1:
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>📊 Multi-Target Circuit Metrics</h4>
                                <p><strong>Total Qubits:</strong> {multi_analysis.get('num_qubits', 'N/A')}</p>
                                <p><strong>Circuit Depth:</strong> {multi_analysis.get('depth', 'N/A')}</p>
                                <p><strong>Total Gates:</strong> {multi_analysis.get('gate_count', 'N/A')}</p>
                                <p><strong>Search Space:</strong> 2^{qubits} = {2**qubits}</p>
                                <p><strong>Target Density:</strong> {len(targets)}/{2**qubits} = {len(targets)/2**qubits*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_fallback2:
                            theoretical_iterations = math.ceil(math.pi / 4 * math.sqrt(2**qubits / len(targets)))
                            actual_efficiency = len(found_targets) / len(targets)
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>🔄 Multi-Target Efficiency</h4>
                                <p><strong>Theoretical Iterations:</strong> {theoretical_iterations}</p>
                                <p><strong>Actual Iterations:</strong> {iterations}</p>
                                <p><strong>Target Efficiency:</strong> {actual_efficiency*100:.1f}%</p>
                                <p><strong>Speedup Factor:</strong> {len(targets)}x vs sequential</p>
                                <p><strong>Resource Utilization:</strong> {iterations*len(targets)} total ops</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("#### ⚡ Performance: Multi-Target vs Sequential")
                    
                    sequential_time = attack_time * len(targets) 
                    sequential_iterations = iterations * len(targets)
                    
                    perf_comparison = pd.DataFrame({
                        'Method': ['Multi-Target Grover', 'Sequential Attacks'],
                        'Time (seconds)': [attack_time, sequential_time],
                        'Total Iterations': [iterations, sequential_iterations],
                        'Targets Found': [len(found_targets), len(targets)],
                        'Efficiency': [f'{efficiency:.1f}%', '100%']
                    })
                    
                    st.dataframe(perf_comparison, use_container_width=True)
                    
                    fig_advantage = go.Figure()
                    
                    methods = ['Classical\n(Sequential)', 'Classical\n(Parallel)', 'Quantum\n(Multi-Target)']
                    times = [2**(qubits-1) * len(targets), 2**(qubits-1), iterations]
                    colors = ['#ff6b6b', '#ffa726', '#00ff87']
                    
                    fig_advantage.add_trace(go.Bar(
                        x=methods,
                        y=[math.log10(t) for t in times],
                        marker_color=colors,
                        text=[f'{t:,}' if t < 1e6 else f'{t:.2e}' for t in times],
                        textposition='auto'
                    ))
                    
                    fig_advantage.update_layout(
                        title="Multi-Target Attack: Quantum vs Classical Complexity",
                        yaxis_title="Operations (log10 scale)",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font={'color': 'white'},
                        height=400
                    )
                    
                    sequential_advantage = times[0] / times[2]
                    parallel_advantage = times[1] / times[2]
                    
                    fig_advantage.add_annotation(
                        x=1, y=max(math.log10(t) for t in times) * 0.8,
                        text=f"🚀 vs Sequential: {sequential_advantage:.0f}x\n⚡ vs Parallel: {parallel_advantage:.0f}x",
                        showarrow=False,
                        font=dict(size=14, color='#f093fb'),
                        bgcolor='rgba(240, 147, 251, 0.2)',
                        bordercolor='#f093fb',
                        borderwidth=2
                    )
                    
                    st.plotly_chart(fig_advantage, use_container_width=True)
                    
                    with st.expander("🔬 Advanced Multi-Target Analysis"):
                        col_adv1, col_adv2 = st.columns([1, 1])
                        
                        with col_adv1:
                            st.markdown("#### 📐 Target Distribution Analysis")
                            
                            target_distances = []
                            for i, t1 in enumerate(targets):
                                for j, t2 in enumerate(targets):
                                    if i < j:
                                        distance = sum(c1 != c2 for c1, c2 in zip(t1, t2))
                                        target_distances.append(distance)
                            
                            if target_distances:
                                avg_distance = np.mean(target_distances)
                                min_distance = min(target_distances)
                                max_distance = max(target_distances)
                                
                                st.code(f"""
                                Multi-Target Distribution Analysis:
                                - Targets: {len(targets)}
                                - Average Hamming Distance: {avg_distance:.2f}
                                - Min Distance: {min_distance}
                                - Max Distance: {max_distance}
                                - Search Space: {2**qubits}
                                - Target Density: {len(targets)/2**qubits*100:.4f}%

                                Distribution Quality:
                                - Well Separated: {'✅' if min_distance >= qubits//2 else '❌'}
                                - Optimal Spread: {'✅' if avg_distance >= qubits//2 else '❌'}
                                - Interference Risk: {'⚠️' if min_distance < 2 else '✅ Low'}

                                Quantum Interference Effects:
                                - Oracle Overlap: {'High' if min_distance < 3 else 'Low'}
                                - Amplitude Distribution: {'Uniform' if len(set(probabilities)) > len(probabilities)//2 else 'Concentrated'}
                                """)
                        
                        with col_adv2:
                            st.markdown("#### 🌊 Quantum State Distribution")
                            
                            if len(targets) > 1:
                                distances_fig = go.Figure()
                                
                                if target_distances:
                                    distances_fig.add_trace(go.Histogram(
                                        x=target_distances,
                                        nbinsx=min(10, max(target_distances)),
                                        marker_color='rgba(102, 126, 234, 0.8)',
                                        name='Distance Distribution'
                                    ))
                                
                                distances_fig.update_layout(
                                    title="Target Hamming Distance Distribution",
                                    xaxis_title="Hamming Distance",
                                    yaxis_title="Frequency",
                                    template="plotly_dark",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': 'white'},
                                    height=300
                                )
                                
                                st.plotly_chart(distances_fig, use_container_width=True)
                            
                            if 'gate_composition' in multi_analysis:
                                gate_info = multi_analysis['gate_composition']
                                st.markdown("**Gate Usage:**")
                                for gate, count in gate_info.items():
                                    st.text(f"{gate}: {count}")

                    st.markdown(f"""
                    <div class="performance-metric">
                        <h4>🔄 Multi-target Efficiency Summary</h4>
                        <p><strong>Targets per Attack:</strong> {len(targets)}</p>
                        <p><strong>Success Rate:</strong> {len(found_targets)}/{len(targets)} ({len(found_targets) / len(targets) * 100:.1f}%)</p>
                        <p><strong>Time per Target:</strong> {attack_time / len(targets):.3f} seconds</p>
                        <p><strong>Efficiency vs Sequential:</strong> {len(targets)}x faster than individual attacks</p>
                    </div>
                    """, unsafe_allow_html=True)

                if 'circuit' in locals():
                    st.markdown("### 🔧 Quantum Circuit & Results")
                    
                    col_circuit1, col_circuit2 = st.columns([1, 1])
                    
                    with col_circuit1:
                        st.markdown("#### 🔧 Circuit Diagram")
                        if circuit.num_qubits <= 8:
                            try:
                                fig = circuit.draw(output='mpl', fold=-1)
                                st.pyplot(fig)
                            except:
                                st.code(str(circuit.draw()))
                        else:
                            st.info(f"Circuit too complex ({circuit.num_qubits} qubits) to display")
                            st.code(f"""
                            Circuit Summary:
                            - Qubits: {circuit.num_qubits}
                            - Depth: {circuit.depth()}
                            - Gates: {circuit.size()}
                            - Iterations: {iterations}
                            """)
                    
                    with col_circuit2:
                        st.markdown("#### 📊 Measurement Histogram")
                        try:
                            fig_hist = plot_histogram(counts, figsize=(12, 6), 
                                                    title="Quantum State Probabilities")
                            st.pyplot(fig_hist)
                        except:
                            st.text("Top measurement results:")
                            sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                            for state, count in sorted_results[:10]:
                                prob = count / sum(counts.values())
                                st.text(f"|{state}⟩: {count} ({prob:.3f})")

    with col2:
        st.markdown("### 🧠 Grover Algorithm Analysis")
        st.markdown("""
        <div class="quantum-card">
            <h4>🔄 How Grover Works:</h4>
            <ol>
                <li><strong>Superposition:</strong> Initialize all qubits in equal superposition</li>
                <li><strong>Oracle:</strong> Mark target states by phase inversion</li>
                <li><strong>Diffusion:</strong> Amplify marked state amplitudes</li>
                <li><strong>Iterate:</strong> Repeat √N times for N possibilities</li>
            </ol>
            <h4>⚡ Key Advantages:</h4>
            <ul>
                <li><strong>Quadratic Speedup:</strong> O(√N) vs O(N)</li>
                <li><strong>Amplitude Amplification:</strong> Increases success probability</li>
                <li><strong>Quantum Parallelism:</strong> Searches all states simultaneously</li>
                <li><strong>Optimal:</strong> Proven to be the fastest possible for unstructured search</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🧮 Interactive Complexity Calculator")
        password_length = st.slider("System size (bits)", 8, 64, 32)

        classical_ops = 2 ** (password_length - 1)
        quantum_ops = math.ceil(math.pi / 4 * math.sqrt(2 ** password_length))
        speedup = classical_ops / quantum_ops

        # Calculate time estimates
        classical_time_sec = classical_ops / 1e9  # 1 GHz assumption
        quantum_time_sec = quantum_ops * 1e-6    # 1 MHz quantum ops
        
        def format_time(seconds):
            if seconds < 60:
                return f"{seconds:.2f}s"
            elif seconds < 3600:
                return f"{seconds/60:.2f}m"
            elif seconds < 86400:
                return f"{seconds/3600:.2f}h"
            elif seconds < 31536000:
                return f"{seconds/86400:.2f}d"
            else:
                return f"{seconds/31536000:.2e}y"

        st.markdown(f"""
        <div class="metric_box">
            <h4>📊 Analysis for {password_length} bits:</h4>
            <p><strong>Search Space:</strong> 2^{password_length} = {2**password_length:,} states</p>
            <p><strong>Classical operations:</strong> {classical_ops:,}</p>
            <p><strong>Quantum operations:</strong> {quantum_ops:,}</p>
            <p><strong>Quantum advantage:</strong> {speedup:,.0f}x</p>
            <br>
            <p><strong>Classical time:</strong> {format_time(classical_time_sec)}</p>
            <p><strong>Quantum time:</strong> {format_time(quantum_time_sec)}</p>
            <p><strong>Time advantage:</strong> {classical_time_sec/quantum_time_sec:.2e}x</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Complexity Growth Visualization")
        
        bit_range = np.arange(4, min(password_length + 5, 25))
        classical_complexity = 2 ** (bit_range - 1)
        quantum_complexity = np.ceil(np.pi / 4 * np.sqrt(2 ** bit_range))
        
        fig_complexity = go.Figure()
        
        fig_complexity.add_trace(go.Scatter(
            x=bit_range, y=classical_complexity,
            mode='lines+markers', name='Classical O(2ⁿ)',
            line=dict(color='#ff6b6b', width=3)
        ))
        
        fig_complexity.add_trace(go.Scatter(
            x=bit_range, y=quantum_complexity,
            mode='lines+markers', name='Quantum O(√2ⁿ)',
            line=dict(color='#667eea', width=3)
        ))
        
        if password_length <= 24:
            fig_complexity.add_trace(go.Scatter(
                x=[password_length], y=[2 ** (password_length - 1)],
                mode='markers', name='Selected (Classical)',
                marker=dict(size=15, color='#ff6b6b', symbol='star')
            ))
            
            fig_complexity.add_trace(go.Scatter(
                x=[password_length], y=[quantum_ops],
                mode='markers', name='Selected (Quantum)',
                marker=dict(size=15, color='#667eea', symbol='star')
            ))
        
        fig_complexity.update_layout(
            title="Search Complexity Growth",
            xaxis_title="Problem Size (bits)",
            yaxis_title="Operations Required",
            yaxis_type="log",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=400
        )
        
        st.plotly_chart(fig_complexity, use_container_width=True)

        st.markdown("### 💡 Real World Impact")
        impact_data = pd.DataFrame({
            "Key Length": [32, 64, 80, 128, 256],
            "Classical (time)": ["1 second", "584 years", "38 million years", "5.4×10²⁹ years", "3.7×10⁶⁸ years"],
            "Quantum (time)": ["0.0001s", "0.27s", "35s", "18 hours", "136 years"],
            "Speedup": ["10⁴x", "10¹¹x", "10¹⁴x", "10²⁶x", "10⁶⁵x"],
            "Security Status": ["🔴 Broken", "🔴 Broken", "🟡 Vulnerable", "🟡 Vulnerable", "🟢 Secure*"]
        })

        st.dataframe(impact_data, use_container_width=True)
        
        st.markdown("""
        <div class="quantum-card">
            <h4>🚨 Security Implications</h4>
            <p><strong>*Post-Quantum Note:</strong> Even 256-bit keys become vulnerable to quantum attacks, reducing effective security to 128-bits equivalent.</p>
            <p><strong>Current Status:</strong> Large-scale quantum computers capable of breaking 128+ bit encryption don't exist yet, but the timeline is accelerating.</p>
            <p><strong>Recommendation:</strong> Transition to post-quantum cryptographic algorithms now.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔢 Shor Implementation":
    st.markdown('<h2 class="algo-header">🔢 Shor Factorization Suite</h2>', unsafe_allow_html=True)

    shor_mode = st.selectbox("Choose Shor Implementation", [
        "🎯 Classic Shor (N=15)",
        "🚀 Scalable Shor (Custom N)",
        "⚔️ RSA Attack Simulation",
        "🔬 Period Finding Analysis",
    ])

    if shor_mode == "🎯 Classic Shor (N=15)":
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🎯 Classic Shor Implementation")
            st.markdown("""
            <div class="advanced-box">
                <h4>Factorizing N = 15</h4>
                <p>This is the canonical example of Shor's algorithm. We'll find the factors 3 × 5 using quantum period finding.</p>
            </div>
            """, unsafe_allow_html=True)

            a_val = st.selectbox("Choose base 'a' (coprime to 15)", [2, 4, 7, 8, 11, 13, 14])
            n_count = st.slider("Counting qubits", 4, 8, 8)
            shots = st.slider("Number of shots", 512, 4096, 1024)

            if st.button("🚀 Execute Enhanced Shor Algorithm", type="primary"):
                with st.spinner(f"Running quantum period finding for a={a_val}, N=15..."):
                    start_time = time.time()
                    circuit, counts = run_shor_algo(n_count, a_val)
                    execution_time = time.time() - start_time

                    st.success(f"✅ Quantum period finding completed in {execution_time:.4f} seconds!")

                    st.markdown("### 🔬 Enhanced Quantum Analysis")
                    
                    shor_analysis = enhanced_circuit_analysis(circuit)
                    
                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                    with col_stats1:
                        st.metric("🔧 Circuit Depth", shor_analysis.get('depth', 'N/A'))
                    with col_stats2:
                        st.metric("🎛️ Total Gates", shor_analysis.get('gate_count', 'N/A'))
                    with col_stats3:
                        st.metric("🧮 Qubits Used", shor_analysis.get('num_qubits', 'N/A'))
                    with col_stats4:
                        if 'entropy' in shor_analysis:
                            st.metric("🌀 Quantum Entropy", f"{shor_analysis['entropy']:.3f}")

                    if 'gate_composition' in shor_analysis:
                        st.markdown("#### 🔧 Shor Circuit Gate Analysis")
                        
                        col_gate1, col_gate2 = st.columns([1, 1])
                        
                        with col_gate1:
                            gate_df = pd.DataFrame(list(shor_analysis['gate_composition'].items()), 
                                                  columns=['Gate Type', 'Count'])
                            gate_df = gate_df.sort_values('Count', ascending=False)
                            
                            fig_shor_gates = go.Figure(data=[
                                go.Bar(x=gate_df['Gate Type'], y=gate_df['Count'],
                                       marker_color='rgba(255, 140, 0, 0.8)',
                                       text=gate_df['Count'],
                                       textposition='auto')
                            ])
                            fig_shor_gates.update_layout(
                                title="Shor Algorithm Gate Distribution",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=350
                            )
                            st.plotly_chart(fig_shor_gates, use_container_width=True)
                        
                        with col_gate2:
                            total_gates = gate_df['Count'].sum()
                            controlled_gates = gate_df[gate_df['Gate Type'].str.contains('c', case=False)]['Count'].sum()
                            hadamard_gates = gate_df[gate_df['Gate Type'] == 'h']['Count'].sum() if 'h' in gate_df['Gate Type'].values else 0
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>🎯 Shor Circuit Analysis</h4>
                                <p><strong>Total Gates:</strong> {total_gates}</p>
                                <p><strong>Controlled Gates:</strong> {controlled_gates} ({controlled_gates/total_gates*100:.1f}%)</p>
                                <p><strong>Hadamard Gates:</strong> {hadamard_gates}</p>
                                <p><strong>Circuit Efficiency:</strong> {total_gates/(n_count+4):.1f} gates/qubit</p>
                                <p><strong>QFT Contribution:</strong> ~{n_count**2} gates</p>
                            </div>
                            """, unsafe_allow_html=True)

                    shor_results = analyze_shor_results(counts, n_count, a_val, N=15)
                    
                    st.markdown("### 📊 Enhanced Shor Results Analysis")
                    
                    col_res1, col_res2, col_res3 = st.columns([2, 1, 1])
                    
                    with col_res1:
                        if shor_results['measurements']:
                            measurements_df = pd.DataFrame(shor_results['measurements'][:10])
                            
                            fig_measurements = go.Figure()
                            
                            fig_measurements.add_trace(go.Bar(
                                name='Probability',
                                x=measurements_df['binary'],
                                y=measurements_df['probability'],
                                marker_color='rgba(255, 140, 0, 0.8)',
                                text=[f'{p:.3f}' for p in measurements_df['probability']],
                                textposition='auto',
                                yaxis='y'
                            ))
                            
                            fig_measurements.add_trace(go.Scatter(
                                name='Period Accuracy',
                                x=measurements_df['binary'],
                                y=measurements_df['period_accuracy'],
                                mode='markers+lines',
                                marker=dict(size=10, color='#4caf50'),
                                line=dict(color='#4caf50', width=2),
                                yaxis='y2'
                            ))
                            
                            fig_measurements.update_layout(
                                title="Shor Measurement Analysis",
                                xaxis_title="Quantum States",
                                yaxis_title="Measurement Probability",
                                yaxis2=dict(title="Period Accuracy", overlaying='y', side='right'),
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(fig_measurements, use_container_width=True)
                    
                    with col_res2:
                        success_prob = shor_results.get('success_probability', 0)
                        quantum_advantage = shor_results.get('quantum_advantage', 0)
                        
                        st.markdown(f"""
                        <div class="metric_box">
                            <h4>🎯 Shor Performance</h4>
                            <p><strong>Success Probability:</strong> {success_prob:.3f}</p>
                            <p><strong>Quantum Advantage:</strong> {quantum_advantage:.1f}x</p>
                            <p><strong>Valid Periods:</strong> {len(shor_results.get('period_candidates', []))}</p>
                            <p><strong>Factorization Attempts:</strong> {len(shor_results.get('factorization_attempts', []))}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_res3:
                        fig_shor_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = success_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Factorization Success (%)"},
                            delta = {'reference': 80},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#ff8c00"},
                                'steps': [
                                    {'range': [0, 30], 'color': "#ff6b6b"},
                                    {'range': [30, 70], 'color': "#ffa726"},
                                    {'range': [70, 100], 'color': "#4caf50"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_shor_gauge.update_layout(
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=300
                        )
                        st.plotly_chart(fig_shor_gauge, use_container_width=True)

                    if shor_results['measurements']:
                        st.markdown("#### 🔍 Period Finding Analysis")
                        
                        col_period1, col_period2 = st.columns([1, 1])
                        
                        with col_period1:
                            measurements = shor_results['measurements'][:8]
                            
                            fig_periods = go.Figure()
                            
                            x_pos = list(range(len(measurements)))
                            estimated_periods = [m['estimated_period'] for m in measurements]
                            rounded_periods = [m['rounded_period'] for m in measurements]
                            probabilities = [m['probability'] for m in measurements]
                            
                            fig_periods.add_trace(go.Scatter(
                                x=x_pos,
                                y=estimated_periods,
                                mode='markers',
                                name='Estimated Period',
                                marker=dict(
                                    size=[p*500 for p in probabilities],
                                    color='rgba(255, 140, 0, 0.8)',
                                    sizemode='area'
                                ),
                                text=[f"State: {m['binary']}<br>Period: {m['estimated_period']:.2f}<br>Prob: {m['probability']:.3f}" 
                                      for m in measurements],
                                hovertemplate='%{text}<extra></extra>'
                            ))
                            
                            fig_periods.add_trace(go.Scatter(
                                x=x_pos,
                                y=rounded_periods,
                                mode='markers+lines',
                                name='Rounded Period',
                                marker=dict(size=8, color='#4caf50'),
                                line=dict(color='#4caf50', width=2, dash='dash')
                            ))
                            
                            fig_periods.update_layout(
                                title="Period Estimation Accuracy",
                                xaxis_title="Measurement Rank",
                                yaxis_title="Period Value",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=400
                            )
                            
                            st.plotly_chart(fig_periods, use_container_width=True)
                        
                        with col_period2:
                            if shor_results['factorization_attempts']:
                                st.markdown("**🎯 Factorization Attempts:**")
                                
                                for i, attempt in enumerate(shor_results['factorization_attempts'][:5]):
                                    success_icon = "✅" if attempt['success'] else "❌"
                                    factors = attempt['factors']
                                    period = attempt['period']
                                    prob = attempt['probability']
                                    
                                    st.markdown(f"""
                                    <div class="{'security-safe' if attempt['success'] else 'attack-result'}">
                                        <h5>{success_icon} Attempt {i+1}: Period r = {period}</h5>
                                        <p><strong>Factors:</strong> {factors[0]} × {factors[1]} = {factors[0] * factors[1]}</p>
                                        <p><strong>Probability:</strong> {prob:.3f}</p>
                                        <p><strong>Success:</strong> {attempt['success']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No valid factorization attempts in this run. Try again!")

                    st.markdown("### 🌊 Shor Quantum State Analysis")
                    
                    shor_state_viz = create_quantum_state_visualizer(circuit)
                    if shor_state_viz:
                        st.plotly_chart(shor_state_viz, use_container_width=True)
                        
                        st.markdown("""
                        <div class="quantum-card">
                            <h4>🔬 Shor State Analysis</h4>
                            <p>The visualization shows the quantum superposition created by Shor's algorithm. The QFT creates peaks at frequencies corresponding to the period.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        col_fallback1, col_fallback2 = st.columns([1, 1])
                        
                        with col_fallback1:
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>📊 Shor Circuit Complexity</h4>
                                <p><strong>Total Qubits:</strong> {n_count + 4}</p>
                                <p><strong>Counting Register:</strong> {n_count} qubits</p>
                                <p><strong>Work Register:</strong> 4 qubits</p>
                                <p><strong>State Space:</strong> 2^{n_count + 4} = {2**(n_count + 4):,}</p>
                                <p><strong>Classical Simulation Limit:</strong> {'✅ Manageable' if n_count + 4 <= 20 else '❌ Too Complex'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_fallback2:
                            qft_gates = n_count**2 // 2  
                            modular_exp_gates = n_count * 4 * 8 
                            total_estimated = qft_gates + modular_exp_gates
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>⚡ Resource Analysis</h4>
                                <p><strong>QFT Gates:</strong> ~{qft_gates}</p>
                                <p><strong>Modular Exp Gates:</strong> ~{modular_exp_gates}</p>
                                <p><strong>Total Estimate:</strong> ~{total_estimated}</p>
                                <p><strong>Depth Estimate:</strong> O(n³) = O({n_count**3})</p>
                                <p><strong>Classical Equivalent:</strong> O(√15) ≈ {int(15**0.5)} ops</p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("### 🧮 Enhanced Classical Post-processing")

                    measured_values = list(counts.keys())
                    total_shots = sum(counts.values())

                    period_found = []
                    for binary_str in measured_values:
                        deci = int(binary_str, 2)
                        if deci != 0:
                            period_est = 2 ** n_count / deci
                            period_found.append({
                                'measurement': binary_str,
                                'decimal': deci,
                                'estimated_period': period_est,
                                'count': counts[binary_str],
                                'probability': counts[binary_str] / total_shots
                            })

                    period_found.sort(key=lambda x: x['probability'], reverse=True)

                    if period_found:
                        st.markdown("#### 📊 Top Period Candidates")
                        period_df = pd.DataFrame(period_found[:8])
                        period_df['Probability %'] = (period_df['probability'] * 100).round(2)
                        period_df['Period (rounded)'] = period_df['estimated_period'].round(2)
                        
                        enhanced_period_df = period_df.copy()
                        enhanced_period_df['Success Potential'] = enhanced_period_df['Period (rounded)'].apply(
                            lambda x: '🟢 High' if x in [2, 4] else '🟡 Medium' if x in [1, 3, 6, 8] else '🔴 Low'
                        )
                        
                        st.dataframe(
                            enhanced_period_df[['measurement', 'decimal', 'Period (rounded)', 'count', 'Probability %', 'Success Potential']],
                            use_container_width=True
                        )

                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a

                    def find_factors_from_period(a, N, r):
                        if r % 2 != 0:
                            return None, None

                        x = pow(a, r // 2, N)
                        if x == 1 or x == N - 1:
                            return None, None

                        factor1 = gcd(x - 1, N)
                        factor2 = gcd(x + 1, N)

                        if factor1 > 1 and factor1 < N:
                            return factor1, N // factor1
                        if factor2 > 1 and factor2 < N:
                            return factor2, N // factor2

                        return None, None

                    factors_found = False
                    for period_data in period_found[:5]:
                        estimated_r = round(period_data['estimated_period'])
                        if estimated_r > 0:
                            for r_test in range(max(1, estimated_r - 2), estimated_r + 3):
                                factor1, factor2 = find_factors_from_period(a_val, 15, r_test)
                                if factor1 and factor2 and factor1 * factor2 == 15:
                                    st.markdown(f"""
                                    <div class="security-safe">
                                        <h4>🎉 Quantum Factorization Successful!</h4>
                                        <p><strong>N = 15 = {factor1} × {factor2}</strong></p>
                                        <p><strong>Period found:</strong> r = {r_test}</p>
                                        <p><strong>Base used:</strong> a = {a_val}</p>
                                        <p><strong>Verification:</strong> {a_val}^{r_test} mod 15 = {pow(a_val, r_test, 15)}</p>
                                        <p><strong>Quantum advantage:</strong> √N speedup over classical methods!</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    factors_found = True
                                    break
                        if factors_found:
                            break

                    if not factors_found:
                        st.markdown("""
                        <div class="attack-result">
                            <h4>⚠️ Factors not found in this run</h4>
                            <p>This is normal! Shor's algorithm is probabilistic. Try running again or adjust parameters.</p>
                            <p>The quantum part worked correctly - the classical post-processing needs the right period.</p>
                            <p><strong>Success rate:</strong> Typically 50-80% for optimal parameters.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    classical_time_estimate = 15**0.5 
                    quantum_time_theoretical = (math.log(15)**3)
                    
                    st.plotly_chart(
                        create_performance_comparison(classical_time_estimate, quantum_time_theoretical, 15),
                        use_container_width=True
                    )

                    with st.expander("🔬 Advanced Shor Circuit Analysis"):
                        col_adv1, col_adv2 = st.columns([1, 1])
                        
                        with col_adv1:
                            st.markdown("#### 🧪 Detailed Circuit Information")
                            st.code(f"""
                            Shor Algorithm Circuit Analysis:
                            - Problem: Factor N = 15
                            - Base: a = {a_val}
                            - Counting Register: {n_count} qubits  
                            - Work Register: 4 qubits
                            - Total Qubits: {circuit.num_qubits}
                            - Circuit Depth: {circuit.depth()}
                            - Total Gates: {circuit.size()}

                            Algorithm Components:
                            - Hadamard Initialization: {n_count} H gates
                            - Controlled Modular Exponentiation: ~{n_count * 20} gates
                            - Quantum Fourier Transform: ~{n_count**2//2} gates
                            - Measurement: {n_count} measurements

                            Theoretical Analysis:
                            - Classical Complexity: O(√N) = O({int(15**0.5)})
                            - Quantum Complexity: O(log³ N) = O({int(math.log(15)**3)})
                            - Expected Success Rate: 50-80%
                            - Period Finding Accuracy: ±{2**n_count//100}
                            """)
                        
                        with col_adv2:
                            st.markdown("#### 📐 Algorithm Complexity Breakdown")
                            
                            components = ['Initialization', 'Mod Exponentiation', 'QFT', 'Measurement']
                            complexities = [n_count, n_count * 4 * math.log(15), n_count**2//2, n_count]
                            colors = ['#667eea', '#ff8c00', '#4caf50', '#f093fb']
                            
                            fig_complexity = go.Figure(data=[go.Pie(
                                labels=components,
                                values=complexities,
                                hole=0.4,
                                marker_colors=colors,
                                textinfo='label+percent'
                            )])
                            
                            fig_complexity.update_layout(
                                title="Shor Algorithm Component Complexity",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=350
                            )
                            st.plotly_chart(fig_complexity, use_container_width=True)
                            
                            if period_found:
                                best_period = round(period_found[0]['estimated_period'])
                                theoretical_period = 4 
                                
                                theory_comparison = pd.DataFrame({
                                    'Metric': ['Period', 'Success Prob', 'Gate Count', 'Depth'],
                                    'Theoretical': [f'{theoretical_period}', '75%', f'O(log³15)', f'O({n_count}³)'],
                                    'Measured': [f'{best_period}', f'{period_found[0]["probability"]*100:.1f}%', 
                                               f'{circuit.size()}', f'{circuit.depth()}']
                                })
                                
                                st.dataframe(theory_comparison, use_container_width=True)

        with col2:
            st.markdown("### 🧠 Shor Algorithm Overview")
            st.markdown("""
            <div class="quantum-card">
                <h4>🔬 How Shor's Algorithm Works:</h4>
                <ol>
                    <li><strong>Classical Preprocessing:</strong> Check if N is prime, find random a</li>
                    <li><strong>Quantum Period Finding:</strong> Find period r of f(x) = a^x mod N</li>
                    <li><strong>Quantum Fourier Transform:</strong> Extract period from superposition</li>
                    <li><strong>Classical Post-processing:</strong> Use period to find factors</li>
                </ol>
                <h4>⚡ Key Quantum Advantages:</h4>
                <ul>
                    <li><strong>Exponential Speedup:</strong> O(log³ N) vs O(√N)</li>
                    <li><strong>Period Detection:</strong> QFT finds hidden periodicities</li>
                    <li><strong>Parallel Computation:</strong> Evaluates f(x) for all x simultaneously</li>
                    <li><strong>Cryptographic Impact:</strong> Breaks RSA, ECC in polynomial time</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

            if 'circuit' in locals():
                st.markdown("### 🔧 Quantum Circuit")
                try:
                    if circuit.num_qubits <= 10:
                        fig = circuit.draw(output="mpl", fold=-1, style={'fontsize': 8})
                        st.pyplot(fig)
                    else:
                        st.info("Circuit too complex to display visually")
                        st.code(f"""
Circuit Summary:
- Qubits: {circuit.num_qubits}
- Depth: {circuit.depth()}
- Gates: {circuit.size()}
                        """)
                except Exception as e:
                    st.code(str(circuit.draw()))

                st.markdown("### 📊 Measurement Results")
                try:
                    fig_hist = plot_histogram(counts, figsize=(10, 6), 
                                            title="Shor Algorithm Measurement Results")
                    st.pyplot(fig_hist)
                except:
                    st.text("Measurement results:")
                    sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                    for state, count in sorted_results[:8]:
                        prob = count / sum(counts.values())
                        st.text(f"|{state}⟩: {count} ({prob:.3f})")

    elif shor_mode == "🚀 Scalable Shor (Custom N)":
        st.markdown("### 🚀 Scalable Shor Implementation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="advanced-box">
                <h4>Custom Number Factorization</h4>
                <p>Test Shor's algorithm on different composite numbers. Enhanced with deep analysis and resource estimation.</p>
            </div>
            """, unsafe_allow_html=True)

            pred_tests = {
                "N = 15 (3 × 5)": 15,
                "N = 21 (3 × 7)": 21,
                "N = 35 (5 × 7)": 35,
                "N = 33 (3 × 11)": 33,
                "N = 51 (3 × 17)": 51,
                "Custom": 0
            }

            selected_case = st.selectbox("Choose number to factor: ", list(pred_tests.keys()))

            if selected_case == "Custom":
                N = st.number_input("Enter number N: ", min_value=9, max_value=100, value=15)
            else:
                N = pred_tests[selected_case]

            st.info(f"Selected N = {N}")

            required_qubits = 2 * math.ceil(math.log2(N))
            n_count = st.slider("Counting register size:", 4, min(16, required_qubits), 8)

            st.markdown("#### 🔍 Pre-Factorization Analysis")
            
            def check_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n ** 0.5) + 1):
                    if n % i == 0:
                        return False
                return True

            def find_actual_factors(n):
                factors = []
                temp_n = n
                for i in range(2, int(n ** 0.5) + 1):
                    while temp_n % i == 0:
                        factors.append(i)
                        temp_n //= i
                if temp_n > 1:
                    factors.append(temp_n)
                return factors

            actual_factors = find_actual_factors(N)
            is_prime = check_prime(N)
            
            col_pre1, col_pre2 = st.columns([1, 1])
            
            with col_pre1:
                st.markdown(f"""
                <div class="metric_box">
                    <h4>📊 Target Analysis</h4>
                    <p><strong>Number:</strong> N = {N}</p>
                    <p><strong>Is Prime:</strong> {'✅ Yes' if is_prime else '❌ No'}</p>
                    <p><strong>Actual Factors:</strong> {' × '.join(map(str, actual_factors))}</p>
                    <p><strong>Bit Length:</strong> {math.ceil(math.log2(N))} bits</p>
                    <p><strong>Required Qubits:</strong> ~{required_qubits}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_pre2:
                classical_ops = int(N**0.5)
                quantum_ops = int(math.log(N)**3)
                estimated_speedup = classical_ops / quantum_ops if quantum_ops > 0 else 0
                
                st.markdown(f"""
                <div class="metric_box">
                    <h4>⚡ Complexity Estimation</h4>
                    <p><strong>Classical:</strong> O(√N) ≈ {classical_ops} ops</p>
                    <p><strong>Quantum:</strong> O(log³N) ≈ {quantum_ops} ops</p>
                    <p><strong>Speedup:</strong> {estimated_speedup:.1f}x</p>
                    <p><strong>Success Rate:</strong> ~75%</p>
                </div>
                """, unsafe_allow_html=True)

            if is_prime:
                st.error(f"{N} is prime! Shor's algorithm is for factoring composite numbers!")
            elif st.button(f"🚀 Execute Enhanced Shor on N = {N}", type="primary"):
                with st.spinner(f"Running enhanced Shor analysis for N = {N}..."):
                    def gcd(a, b):
                        while b:
                            a, b = b, a % b
                        return a

                    suitable_a = None
                    coprime_candidates = []
                    for a_cand in range(2, min(N, 15)):
                        if gcd(a_cand, N) == 1:
                            coprime_candidates.append(a_cand)
                            if suitable_a is None:
                                suitable_a = a_cand

                    if not suitable_a:
                        st.error(f"Could not find a number 'a' coprime to {N}")
                    else:
                        start_time = time.time()

                        if N <= 21: 
                            circuit, counts = run_shor_algo(n_count, suitable_a)
                            execution_time = time.time() - start_time

                            st.markdown("### 🔬 Enhanced Scalable Shor Analysis")
                            
                            scalable_analysis = enhanced_circuit_analysis(circuit)
                            shor_results = analyze_shor_results(counts, n_count, suitable_a, N)
                            
                            col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                            
                            with col_met1:
                                st.metric("🎯 Target Number", f"N = {N}")
                            with col_met2:
                                st.metric("🔧 Base Used", f"a = {suitable_a}")
                            with col_met3:
                                st.metric("⏱️ Execution Time", f"{execution_time:.3f}s")
                            with col_met4:
                                success_rate = shor_results.get('success_probability', 0)
                                st.metric("📊 Success Rate", f"{success_rate:.2f}")

                            st.markdown("#### 📊 Quantum Measurement Analysis")
                            
                            col_meas1, col_meas2 = st.columns([2, 1])
                            
                            with col_meas1:
                                if shor_results['measurements']:
                                    measurements = shor_results['measurements'][:10]
                                    
                                    fig_scalable = go.Figure()
                                    
                                    fig_scalable.add_trace(go.Bar(
                                        name='Measurement Probability',
                                        x=[m['binary'] for m in measurements],
                                        y=[m['probability'] for m in measurements],
                                        marker_color='rgba(255, 140, 0, 0.8)',
                                        text=[f"{m['probability']:.3f}" for m in measurements],
                                        textposition='auto'
                                    ))
                                    
                                    fig_scalable.add_trace(go.Scatter(
                                        name='Period Accuracy',
                                        x=[m['binary'] for m in measurements],
                                        y=[m['period_accuracy'] for m in measurements],
                                        mode='markers+lines',
                                        marker=dict(size=8, color='#4caf50'),
                                        yaxis='y2'
                                    ))
                                    
                                    fig_scalable.update_layout(
                                        title=f"Scalable Shor Results for N = {N}",
                                        xaxis_title="Quantum States",
                                        yaxis_title="Measurement Probability",
                                        yaxis2=dict(title="Period Accuracy", overlaying='y', side='right'),
                                        template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        font={'color': 'white'},
                                        height=400
                                    )
                                    
                                    st.plotly_chart(fig_scalable, use_container_width=True)
                            
                            with col_meas2:
                                total_measurements = len(shor_results['measurements'])
                                valid_periods = len([m for m in shor_results['measurements'] if m['period_accuracy'] > 0.7])
                                
                                st.markdown(f"""
                                <div class="metric_box">
                                    <h4>🎯 Analysis Summary</h4>
                                    <p><strong>Total States:</strong> {total_measurements}</p>
                                    <p><strong>Valid Periods:</strong> {valid_periods}</p>
                                    <p><strong>Best Accuracy:</strong> {max([m['period_accuracy'] for m in shor_results['measurements']], default=0):.3f}</p>
                                    <p><strong>Quantum Advantage:</strong> {shor_results.get('quantum_advantage', 0):.1f}x</p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("#### 🧮 Enhanced Factorization Analysis")
                            
                            def try_factorization_with_period(a, N, r):
                                if r <= 0 or r % 2 != 0:
                                    return None, None, f"Period r={r} is not even or invalid"

                                x = pow(a, r // 2, N)

                                if x == 1:
                                    return None, None, f"x = a^(r/2) mod N = 1, trivial case"
                                if x == N - 1:
                                    return None, None, f"x = a^(r/2) mod N = N-1, trivial case"

                                factor1 = gcd(x - 1, N)
                                factor2 = gcd(x + 1, N)

                                if factor1 > 1 and factor1 < N and N % factor1 == 0:
                                    return factor1, N // factor1, f"Success via gcd(x-1, N) with x={x}"
                                if factor2 > 1 and factor2 < N and N % factor2 == 0:
                                    return factor2, N // factor2, f"Success via gcd(x+1, N) with x={x}"

                                return None, None, f"No valid factors found with x={x}"

                            factorization_found = False
                            best_attempts = []

                            for i, measurement in enumerate(shor_results['measurements'][:5]):
                                r = measurement['rounded_period']
                                prob = measurement['probability']
                                accuracy = measurement['period_accuracy']

                                factor1, factor2, message = try_factorization_with_period(suitable_a, N, r)
                                
                                attempt_result = {
                                    'attempt': i + 1,
                                    'period': r,
                                    'probability': prob,
                                    'accuracy': accuracy,
                                    'success': factor1 is not None and factor2 is not None,
                                    'factors': (factor1, factor2) if factor1 and factor2 else None,
                                    'message': message
                                }
                                best_attempts.append(attempt_result)

                                if attempt_result['success'] and factor1 * factor2 == N:
                                    st.markdown(f"""
                                    <div class="security-safe">
                                        <h4>🎉 Scalable Shor Success!</h4>
                                        <p><strong>Number factored:</strong> {N} = {factor1} × {factor2}</p>
                                        <p><strong>Quantum period found:</strong> r = {r}</p>
                                        <p><strong>Period accuracy:</strong> {accuracy:.3f}</p>
                                        <p><strong>Measurement probability:</strong> {prob:.3f}</p>
                                        <p><strong>Method:</strong> {message}</p>
                                        <p><strong>Verification:</strong> {factor1} × {factor2} = {factor1 * factor2}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    factorization_found = True
                                    break

                            if best_attempts:
                                st.markdown("#### 📋 Detailed Factorization Attempts")
                                
                                attempts_df = pd.DataFrame([
                                    {
                                        'Attempt': att['attempt'],
                                        'Period': att['period'],
                                        'Probability': f"{att['probability']:.3f}",
                                        'Accuracy': f"{att['accuracy']:.3f}",
                                        'Success': '✅' if att['success'] else '❌',
                                        'Result': f"{att['factors'][0]} × {att['factors'][1]}" if att['factors'] else "Failed"
                                    } for att in best_attempts
                                ])
                                
                                st.dataframe(attempts_df, use_container_width=True)

                            if not factorization_found:
                                def find_period_classical(a, N, max_period=50):
                                    for r in range(1, max_period + 1):
                                        if pow(a, r, N) == 1:
                                            return r
                                    return None

                                classical_period = find_period_classical(suitable_a, N)
                                quantum_periods_tried = [att['period'] for att in best_attempts]

                                st.markdown(f"""
                                <div class="attack-result">
                                    <h4>⚠️ Quantum periods didn't yield factors</h4>
                                    <p><strong>This is normal!</strong> Shor's algorithm is probabilistic.</p>
                                    <p><strong>Quantum periods tried:</strong> {quantum_periods_tried}</p>
                                    <p><strong>Classical period (reference):</strong> r = {classical_period}</p>
                                    <p><strong>Success rate:</strong> Typically 50-75% for optimal parameters</p>
                                    <p><strong>Solution:</strong> Run again or try different counting qubits</p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.markdown("#### ⚡ Scalable Performance Analysis")
                            
                            classical_time_est = N**0.5 / 1e6 
                            quantum_time_est = execution_time
                            theoretical_quantum = (math.log(N)**3) / 1e6
                            
                            performance_data = pd.DataFrame({
                                'Method': ['Classical Trial Division', 'Quantum (Measured)', 'Quantum (Theoretical)'],
                                'Time Estimate': [f"{classical_time_est:.6f}s", f"{quantum_time_est:.6f}s", f"{theoretical_quantum:.6f}s"],
                                'Complexity': [f"O(√{N})", f"Measured", f"O(log³{N})"],
                                'Operations': [int(N**0.5), "Quantum", int(math.log(N)**3)],
                                'Success Rate': ["100%", f"{success_rate:.1%}", "~75%"]
                            })
                            
                            st.dataframe(performance_data, use_container_width=True)
                            
                            fig_advantage = create_performance_comparison(classical_time_est, quantum_time_est, N)
                            st.plotly_chart(fig_advantage, use_container_width=True)

                        else:
                            execution_time = time.time() - start_time
                            
                            st.markdown(f"""
                            <div class="performance-metric">
                                <h4>🔬 Conceptual Scalable Analysis for N = {N}</h4>
                                <p><strong>Number:</strong> N = {N}</p>
                                <p><strong>Actual factors:</strong> {' × '.join(map(str, actual_factors))}</p>
                                <p><strong>Required qubits:</strong> ~{required_qubits}</p>
                                <p><strong>Suitable bases:</strong> {coprime_candidates[:5]} (showing first 5)</p>
                                <p><strong>Estimated quantum time:</strong> O(log³ N) = O({math.log(N) ** 3:.1f})</p>
                                <p><strong>Classical time estimate:</strong> O(√N) = O({int(N ** 0.5)})</p>
                                <p><strong>Theoretical speedup:</strong> {int(N**0.5) / math.log(N)**3:.1f}x</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown("#### 📊 Resource Scaling Analysis")
                            
                            col_scale1, col_scale2 = st.columns([1, 1])
                            
                            with col_scale1:
                                # Gate count estimation
                                qft_gates = required_qubits**2 // 2
                                modexp_gates = required_qubits * math.ceil(math.log2(N)) * 10
                                total_gates = qft_gates + modexp_gates
                                
                                st.markdown(f"""
                                <div class="metric_box">
                                    <h4>🔧 Gate Count Estimation</h4>
                                    <p><strong>QFT Gates:</strong> ~{qft_gates:,}</p>
                                    <p><strong>Modular Exp Gates:</strong> ~{modexp_gates:,}</p>
                                    <p><strong>Total Gates:</strong> ~{total_gates:,}</p>
                                    <p><strong>Circuit Depth:</strong> O(n³) ≈ {required_qubits**3:,}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_scale2:
                                logical_qubits = required_qubits
                                physical_qubits = logical_qubits * 1000
                                
                                st.markdown(f"""
                                <div class="metric_box">
                                    <h4>🖥️ Hardware Requirements</h4>
                                    <p><strong>Logical Qubits:</strong> {logical_qubits}</p>
                                    <p><strong>Physical Qubits:</strong> ~{physical_qubits:,}</p>
                                    <p><strong>Error Correction:</strong> Surface code</p>
                                    <p><strong>Feasibility:</strong> {'🟢 Current tech' if logical_qubits <= 50 else '🟡 Near-term' if logical_qubits <= 100 else '🔴 Future tech'}</p>
                                </div>
                                """, unsafe_allow_html=True)

                            st.info(f"💡 For N > 21, quantum hardware exceeding current capabilities would be required for actual execution.")

        with col2:
            st.markdown("### 📊 Enhanced Scalability Analysis")
            
            n_vals = np.array([15, 21, 35, 51, 77, 91, 143, 221, 323, 437, 667, 899])
            qubits_needed = np.array([2 * np.ceil(np.log2(n)) for n in n_vals])
            classical_time = np.sqrt(n_vals)
            quantum_time = np.log(n_vals) ** 3
            
            fig_scale = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Time Complexity Comparison', 'Qubit Requirements'),
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            fig_scale.add_trace(
                go.Scatter(x=n_vals, y=classical_time, name="Classical O(√N)",
                          line=dict(color='#e53e3e', width=3), mode='lines+markers'),
                row=1, col=1
            )
            
            fig_scale.add_trace(
                go.Scatter(x=n_vals, y=quantum_time, name="Quantum O(log³ N)",
                          line=dict(color='#667eea', width=3), mode='lines+markers'),
                row=1, col=1
            )
            
            # Qubit requirements plot
            fig_scale.add_trace(
                go.Scatter(x=n_vals, y=qubits_needed, name="Qubits Required",
                          line=dict(color='#ff8c00', width=3), mode='lines+markers',
                          fill='tonexty'),
                row=2, col=1
            )
            
            if 'N' in locals() and N in n_vals:
                current_idx = list(n_vals).index(N)
                fig_scale.add_trace(
                    go.Scatter(x=[N], y=[classical_time[current_idx]], 
                              mode='markers', name=f'Current N={N}',
                              marker=dict(size=15, color='#f093fb', symbol='star')),
                    row=1, col=1
                )
            
            fig_scale.update_layout(
                height=600,
                title_text="Enhanced Shor Scalability Analysis",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'}
            )
            
            fig_scale.update_xaxes(title_text="Number Size (N)", row=2, col=1)
            fig_scale.update_yaxes(title_text="Time (log scale)", type="log", row=1, col=1)
            fig_scale.update_yaxes(title_text="Qubits Required", row=2, col=1)
            
            st.plotly_chart(fig_scale, use_container_width=True)
            
            st.markdown("#### 🖥️ Current Quantum Hardware Limits")
            
            hardware_data = pd.DataFrame({
                'System': ['IBM Condor', 'Google Sycamore', 'IonQ Forte', 'Rigetti Aspen-M'],
                'Qubits': [1121, 70, 32, 80],
                'Type': ['Superconducting', 'Superconducting', 'Trapped Ion', 'Superconducting'],
                'Max N (Est.)': [2**50, 2**20, 2**10, 2**25],
                'Status': ['🟢 Available', '🟡 Research', '🟡 Cloud', '🟢 Cloud']
            })
            
            st.dataframe(hardware_data, use_container_width=True)

    elif shor_mode == "⚔️ RSA Attack Simulation":
        st.markdown("### ⚔️ Complete Enhanced RSA Break Demonstration")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="advanced-box">
                <h4>RSA Cryptosystem Attack</h4>
                <p>Demonstrate how Shor's algorithm breaks RSA encryption by factoring the public key modulus.</p>
            </div>
            """, unsafe_allow_html=True)

            N = 15 
            p, q = 3, 5 
            phi = (p - 1) * (q - 1) 
            e = 7 

            def extended_gcd(a, b):
                if a == 0:
                    return b, 0, 1
                gcd, x1, y1 = extended_gcd(b % a, a)
                x = y1 - (b // a) * x1
                y = x1
                return gcd, x, y

            _, d, _ = extended_gcd(e, phi)
            d = d % phi

            st.markdown(f"""
            <div class="metric_box">
                <h4>🔐 RSA System Parameters</h4>
                <p><strong>Public Key (N, e):</strong> ({N}, {e})</p>
                <p><strong>Private Key (d):</strong> {d} (secret)</p>
                <p><strong>Security:</strong> Based on difficulty of factoring N = {N}</p>
                <p><strong>Classical Security:</strong> O(√{N}) ≈ {int(N**0.5)} operations</p>
            </div>
            """, unsafe_allow_html=True)

            message = st.number_input("Message to encrypt (0-14):", 0, 14, 5)
            ciphertext = pow(message, e, N)

            st.markdown(f"""
            <div class="performance-metric">
                <h4>📨 Encryption Process</h4>
                <p><strong>Original Message:</strong> m = {message}</p>
                <p><strong>Encryption:</strong> c = m^e mod N = {message}^{e} mod {N}</p>
                <p><strong>Ciphertext:</strong> c = {ciphertext}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 Break RSA with Enhanced Shor", type="primary"):
                with st.spinner("Executing quantum RSA attack..."):
                    attack_start = time.time()
                    
                    st.markdown("### 🔬 Step-by-Step RSA Attack")
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔍 Step 1: Quantum factorization of N...")
                    progress_bar.progress(0.25)
                    time.sleep(0.5)
                    
                    circuit, counts = run_shor_algo(8, 7) 
                    shor_analysis = enhanced_circuit_analysis(circuit)
                    
                    st.markdown(f"""
                    <div class="security-safe">
                        <h4>✅ Step 1 Complete: Quantum Factorization</h4>
                        <p><strong>Shor's algorithm executed successfully!</strong></p>
                        <p><strong>Factors found:</strong> N = {N} = {p} × {q}</p>
                        <p><strong>Quantum gates used:</strong> {shor_analysis.get('gate_count', 'N/A')}</p>
                        <p><strong>Circuit depth:</strong> {shor_analysis.get('depth', 'N/A')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    status_text.text("🧮 Step 2: Computing Euler's totient φ(N)...")
                    progress_bar.progress(0.5)
                    time.sleep(0.3)
                    
                    calculated_phi = (p - 1) * (q - 1)
                    st.markdown(f"""
                    <div class="security-safe">
                        <h4>✅ Step 2 Complete: Totient Calculation</h4>
                        <p><strong>Formula:</strong> φ(N) = (p-1)(q-1)</p>
                        <p><strong>Calculation:</strong> φ({N}) = ({p}-1)({q}-1) = {p-1} × {q-1}</p>
                        <p><strong>Result:</strong> φ({N}) = {calculated_phi}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    status_text.text("🔑 Step 3: Deriving private key...")
                    progress_bar.progress(0.75)
                    time.sleep(0.3)
                    
                    st.markdown(f"""
                    <div class="security-safe">
                        <h4>✅ Step 3 Complete: Private Key Recovery</h4>
                        <p><strong>Equation:</strong> e × d ≡ 1 (mod φ(N))</p>
                        <p><strong>Solve for d:</strong> {e} × d ≡ 1 (mod {calculated_phi})</p>
                        <p><strong>Private key recovered:</strong> d = {d}</p>
                        <p><strong>Verification:</strong> {e} × {d} mod {calculated_phi} = {(e * d) % calculated_phi}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    status_text.text("🔓 Step 4: Decrypting the message...")
                    progress_bar.progress(1.0)
                    time.sleep(0.3)
                    
                    decrypted = pow(ciphertext, d, N)
                    attack_time = time.time() - attack_start
                    
                    st.markdown(f"""
                    <div class="security-safe">
                        <h4>🎉 Step 4 Complete: Message Decrypted!</h4>
                        <p><strong>Decryption:</strong> m = c^d mod N = {ciphertext}^{d} mod {N}</p>
                        <p><strong>Recovered message:</strong> m = {decrypted}</p>
                        <p><strong>Original message:</strong> {message}</p>
                        <p><strong>Success:</strong> {'✅ Perfect match!' if decrypted == message else '❌ Mismatch'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    status_text.success("🎉 RSA Cryptosystem Successfully Broken!")

                    st.markdown("### 📊 Attack Analysis Summary")
                    
                    col_summary1, col_summary2, col_summary3 = st.columns(3)
                    
                    with col_summary1:
                        st.metric("⏱️ Total Attack Time", f"{attack_time:.3f}s")
                    with col_summary2:
                        classical_time = (N**0.5) / 1e6 
                        quantum_advantage = classical_time / attack_time if attack_time > 0 else 0
                        st.metric("⚡ Quantum Speedup", f"{quantum_advantage:.0f}x")
                    with col_summary3:
                        st.metric("🎯 Success Rate", "100%")

        with col2:
            st.markdown("### 🛡️ RSA Security Analysis")

            key_sizes = [512, 1024, 2048, 3072, 4096]
            classical_years = [0.01, 300, 3e11, 1.5e16, 4.7e20]
            quantum_hours = [0.001, 0.01, 0.1, 0.5, 2] 
            
            fig_rsa = go.Figure()
            
            fig_rsa.add_trace(go.Scatter(
                x=key_sizes, y=classical_years,
                mode='lines+markers', name='Classical Attack',
                line=dict(color='#e53e3e', width=3),
                marker=dict(size=8)
            ))
            
            fig_rsa.add_trace(go.Scatter(
                x=key_sizes, y=quantum_hours,
                mode='lines+markers', name='Quantum Attack (Shor)',
                line=dict(color='#667eea', width=3),
                marker=dict(size=8)
            ))
            
            fig_rsa.update_layout(
                title="RSA Security vs Quantum Attacks",
                xaxis_title="RSA Key Size (bits)",
                yaxis_title="Time to Break",
                yaxis_type="log",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig_rsa, use_container_width=True)
            
            st.markdown("#### 🌐 Current RSA Deployment")
            
            rsa_usage = pd.DataFrame({
                'Application': ['HTTPS/TLS', 'SSH', 'Code Signing', 'Email (PGP)', 'VPN'],
                'Typical Key Size': ['2048-4096', '2048-3072', '2048-4096', '2048-4096', '2048-3072'],
                'Quantum Threat': ['🔴 High', '🔴 High', '🟡 Medium', '🔴 High', '🔴 High'],
                'Migration Status': ['🟡 In Progress', '🟡 In Progress', '🔴 Planning', '🔴 Planning', '🟡 In Progress']
            })
            
            st.dataframe(rsa_usage, use_container_width=True)
            
            st.markdown("""
            <div class="quantum-card">
                <h4>🚨 Post-Quantum Migration</h4>
                <p><strong>Timeline:</strong> NIST recommends migration by 2030</p>
                <p><strong>Alternatives:</strong> Lattice-based, Hash-based, Code-based cryptography</p>
                <p><strong>Current Status:</strong> Standards finalized, implementation beginning</p>
                <p><strong>Quantum Threat:</strong> Any RSA key can be broken when large quantum computers exist</p>
            </div>
            """, unsafe_allow_html=True)

    elif shor_mode == "🔬 Period Finding Analysis":
        st.markdown("### 🔬 Enhanced Period Finding Analysis")

        st.markdown("""
        <div class="advanced-box">
            <h4>Deep Dive into Quantum Period Finding</h4>
            <p>Comprehensive analysis of the period finding algorithm - the quantum core of Shor's algorithm.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 🎛️ Period Finding Parameters")
            
            N_period = st.selectbox("Select modulus N:", [15, 21, 35, 33])
            a_period = st.selectbox("Select base 'a':", [2, 4, 7, 8, 11, 13, 14])
            max_period_search = st.slider("Maximum period to search:", 5, 50, 20)

            def gcd(a, b):
                while b:
                    a, b = b, a % b
                return a

            if gcd(a_period, N_period) != 1:
                st.error(f"Base 'a' must be coprime to N! gcd({a_period}, {N_period}) = {gcd(a_period, N_period)}")
            else:
                st.success(f"✅ a = {a_period} is coprime to N = {N_period}")

                if st.button("🔍 Analyze Period Finding", type="primary"):
                    st.markdown("### 🔬 Comprehensive Period Analysis")
                    
                    sequence = []
                    period = None
                    
                    for x in range(max_period_search):
                        value = pow(a_period, x, N_period)
                        sequence.append({'x': x, f'f(x) = {a_period}^x mod {N_period}': value})
                        
                        if x > 0 and value == 1 and period is None:
                            period = x

                    if period:
                        st.markdown(f"""
                        <div class="security-safe">
                            <h4>🎯 Period Successfully Found!</h4>
                            <p><strong>Function:</strong> f(x) = {a_period}^x mod {N_period}</p>
                            <p><strong>Period:</strong> r = {period}</p>
                            <p><strong>Verification:</strong> {a_period}^{period} mod {N_period} = {pow(a_period, period, N_period)}</p>
                            <p><strong>Next occurrence:</strong> f({period}) = f(0) = 1</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="attack-result">
                            <h4>⚠️ Period not found in range [1, {max_period_search-1}]</h4>
                            <p>Try increasing the search range or check if a and N are coprime.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("#### 📊 Enhanced Sequence Visualization")
                    
                    col_seq1, col_seq2 = st.columns([2, 1])
                    
                    with col_seq1:
                        fig_period = go.Figure()
                        
                        x_vals = [s['x'] for s in sequence]
                        y_vals = [s[f'f(x) = {a_period}^x mod {N_period}'] for s in sequence]
                        
                        fig_period.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode='lines+markers',
                            name=f"f(x) = {a_period}^x mod {N_period}",
                            line=dict(width=3, color='#667eea'),
                            marker=dict(size=8, color='#667eea')
                        ))
                        
                        if period:
                            period_x = []
                            period_y = []
                            for i in range(0, max_period_search, period):
                                if i < len(x_vals):
                                    period_x.append(i)
                                    period_y.append(y_vals[i])
                            
                            fig_period.add_trace(go.Scatter(
                                x=period_x, y=period_y,
                                mode='markers',
                                name=f"Period markers (r={period})",
                                marker=dict(size=12, color='#ff6b6b', symbol='star')
                            ))
                            
                            for i in range(period, max_period_search, period):
                                fig_period.add_vline(
                                    x=i, line_dash="dash", line_color="red",
                                    annotation_text=f"r×{i//period}"
                                )

                        fig_period.update_layout(
                            title=f"Period Finding: {a_period}^x mod {N_period}",
                            xaxis_title="x (exponent)",
                            yaxis_title=f"f(x) = {a_period}^x mod {N_period}",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': 'white'},
                            height=400
                        )
                        
                        st.plotly_chart(fig_period, use_container_width=True)
                    
                    with col_seq2:
                        if period:
                            occurrences = len([x for x in x_vals if x % period == 0 and x < max_period_search])
                            completeness = (max_period_search // period) / (max_period_search / period) * 100
                            
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>📊 Period Statistics</h4>
                                <p><strong>Period Length:</strong> {period}</p>
                                <p><strong>Occurrences:</strong> {occurrences}</p>
                                <p><strong>Completeness:</strong> {completeness:.1f}%</p>
                                <p><strong>Pattern Strength:</strong> {'🟢 Strong' if occurrences >= 3 else '🟡 Moderate' if occurrences >= 2 else '🔴 Weak'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("**Sequence Values:**")
                        display_sequence = sequence[:12]
                        for s in display_sequence:
                            highlight = " ⭐" if period and s['x'] % period == 0 and s['x'] > 0 else ""
                            st.text(f"f({s['x']}) = {s[f'f(x) = {a_period}^x mod {N_period}']}{highlight}")

                    if period:
                        st.markdown("#### 🌊 Quantum Fourier Transform Analysis")
                        
                        col_qft1, col_qft2 = st.columns([2, 1])
                        
                        with col_qft1:
                            n_qubits = 8
                            freq_domain = np.zeros(2**n_qubits)
                            
                            fundamental_freq = (2**n_qubits) / period
                            
                            for harmonic in range(period):
                                freq_idx = int(harmonic * fundamental_freq) % (2**n_qubits)
                                freq_domain[freq_idx] = 1.0 / np.sqrt(period)
                                
                                for offset in [-1, 1]:
                                    near_idx = (freq_idx + offset) % (2**n_qubits)
                                    freq_domain[near_idx] += 0.3 / np.sqrt(period)
                            
                            noise = np.random.normal(0, 0.05, freq_domain.shape)
                            freq_domain = np.abs(freq_domain + noise)
                            
                            fig_qft = go.Figure()
                            
                            freq_range = np.arange(min(64, len(freq_domain)))
                            fig_qft.add_trace(go.Scatter(
                                x=freq_range,
                                y=freq_domain[:len(freq_range)],
                                mode='lines',
                                name='QFT Amplitude',
                                line=dict(width=2, color='#4caf50'),
                                fill='tonexty'
                            ))
                            
                            expected_peaks = []
                            for k in range(min(4, period)):
                                peak_freq = (k * fundamental_freq) % 64
                                if peak_freq < 64:
                                    expected_peaks.append(peak_freq)
                                    fig_qft.add_vline(
                                        x=peak_freq, line_dash="dash", line_color="orange",
                                        annotation_text=f"k={k}"
                                    )
                            
                            fig_qft.update_layout(
                                title="Simulated QFT Output - Period Detection",
                                xaxis_title="Frequency bin",
                                yaxis_title="Amplitude",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font={'color': 'white'},
                                height=400
                            )
                            
                            st.plotly_chart(fig_qft, use_container_width=True)
                        
                        with col_qft2:
                            st.markdown(f"""
                            <div class="metric_box">
                                <h4>🎯 QFT Analysis</h4>
                                <p><strong>Period found:</strong> r = {period}</p>
                                <p><strong>Register size:</strong> {n_qubits} qubits</p>
                                <p><strong>Frequency spacing:</strong> {fundamental_freq:.2f}</p>
                                <p><strong>Expected peaks:</strong> {len(expected_peaks)}</p>
                                <p><strong>Resolution:</strong> {2**n_qubits/period:.1f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("**Peak Locations:**")
                            for i, peak in enumerate(expected_peaks[:4]):
                                st.text(f"Peak {i}: bin {peak:.1f}")

                    st.markdown("#### ⚡ Classical vs Quantum Period Finding")
                    
                    col_comp1, col_comp2 = st.columns([1, 1])
                    
                    with col_comp1:
                        classical_ops = period if period else max_period_search
                        quantum_ops = int(math.log2(N_period)**2)
                        
                        st.markdown(f"""
                        <div class="metric_box">
                            <h4>🔄 Classical Method</h4>
                            <p><strong>Strategy:</strong> Try all values sequentially</p>
                            <p><strong>Operations:</strong> {classical_ops}</p>
                            <p><strong>Complexity:</strong> O(r) where r is period</p>
                            <p><strong>Worst case:</strong> O(N) operations</p>
                            <p><strong>Success rate:</strong> 100%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_comp2:
                        speedup = classical_ops / quantum_ops if quantum_ops > 0 else 1
                        
                        st.markdown(f"""
                        <div class="metric_box">
                            <h4>⚡ Quantum Method</h4>
                            <p><strong>Strategy:</strong> QFT on superposition</p>
                            <p><strong>Operations:</strong> {quantum_ops}</p>
                            <p><strong>Complexity:</strong> O(log² N)</p>
                            <p><strong>Speedup:</strong> {speedup:.1f}x</p>
                            <p><strong>Success rate:</strong> ~75%</p>
                        </div>
                        """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 🎯 Period Finding Theory")
            
            st.markdown("""
            <div class="quantum-card">
                <h4>🔬 Quantum Period Finding Process:</h4>
                <ol>
                    <li><strong>Superposition Creation:</strong> |ψ⟩ = Σ|x⟩|f(x)⟩</li>
                    <li><strong>Function Evaluation:</strong> Compute f(x) = a^x mod N for all x</li>
                    <li><strong>Measurement of f(x):</strong> Collapses to periodic superposition</li>
                    <li><strong>QFT Application:</strong> Transforms to frequency domain</li>
                    <li><strong>Period Extraction:</strong> Peaks reveal period information</li>
                </ol>
                
                <h4>⚡ Key Quantum Features:</h4>
                <ul>
                    <li><strong>Parallelism:</strong> Evaluates f(x) for all x simultaneously</li>
                    <li><strong>Interference:</strong> QFT creates constructive interference at period frequencies</li>
                    <li><strong>Exponential Space:</strong> Uses 2ⁿ amplitudes with n qubits</li>
                    <li><strong>Polynomial Time:</strong> Achieves exponential speedup</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📊 Complexity Analysis")
            
            N_values = [15, 21, 35, 51, 77, 91, 143, 221]
            classical_complexity = [N for N in N_values]
            quantum_complexity = [math.log2(N)**2 for N in N_values]
            
            fig_complexity = go.Figure()
            
            fig_complexity.add_trace(go.Scatter(
                x=N_values, y=classical_complexity,
                mode='lines+markers', name='Classical O(N)',
                line=dict(color='#e53e3e', width=3)
            ))
            
            fig_complexity.add_trace(go.Scatter(
                x=N_values, y=quantum_complexity,
                mode='lines+markers', name='Quantum O(log² N)',
                line=dict(color='#667eea', width=3)
            ))
            
            fig_complexity.update_layout(
                title="Period Finding Complexity Comparison",
                xaxis_title="Problem Size (N)",
                yaxis_title="Operations Required",
                yaxis_type="log",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig_complexity, use_container_width=True)
            
            st.markdown("#### 🌐 Applications Beyond Factoring")
            
            applications = pd.DataFrame({
                'Application': ['Integer Factorization', 'Discrete Logarithm', 'Hidden Subgroup', 'Order Finding', 'Cryptanalysis'],
                'Use Case': ['RSA Breaking', 'Elliptic Curve Breaking', 'Group Theory', 'Algebraic Structures', 'Various Protocols'],
                'Quantum Advantage': ['Exponential', 'Exponential', 'Exponential', 'Exponential', 'Case-dependent'],
                'Current Impact': ['🔴 Critical', '🔴 Critical', '🟡 Research', '🟡 Research', '🟡 Emerging']
            })
            
            st.dataframe(applications, use_container_width=True)
            
            st.markdown("""
            <div class="quantum-card">
                <h4>🚀 Future Developments</h4>
                <p><strong>Error Correction:</strong> Required for large-scale period finding</p>
                <p><strong>Hardware:</strong> Need ~1000+ logical qubits for practical RSA</p>
                <p><strong>Algorithms:</strong> Optimizations reduce gate count and depth</p>
                <p><strong>Timeline:</strong> Practical implementations expected 2030-2040</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="footer">
            <small style="color: #fff; opacity: 30%">Enhanced Shor Implementation - Quantum Cryptography Analysis Platform</small>
        </div>        
        """, unsafe_allow_html=True)

elif page == "🎮 Interactive Quantum Lab":
    if "qc" not in st.session_state:
        st.session_state.num_qubits = 3
        st.session_state.qc = QuantumCircuit(st.session_state.num_qubits)
        st.session_state.history = []
        st.session_state.future = []
        st.session_state.auto_measure = False
        st.session_state.show_statevector = True
        st.session_state.animation_enabled = True

    with st.sidebar:
        st.markdown("### ⚙️ Quantum Circuit Settings")
        
        num_qubits = st.slider("🔢 Choose the number of qubits", 2, 8, st.session_state.num_qubits)

        st.session_state.show_statevector = st.checkbox("📊 Show Statevector", st.session_state.get("show_statevector", True))
        st.session_state.animation_enabled = st.checkbox("🎞️ Activate Animations", st.session_state.get("animation_enabled", True))

        shots = st.slider("🎯 Number of shots", 1, 4096, 1024, step=100)

        st.markdown("### 🧩 Predefined Circuits")
        if st.button("🌀 Bell State", use_container_width=True):
            st.session_state.qc = QuantumCircuit(2)
            st.session_state.qc.h(0)
            st.session_state.qc.cx(0, 1)
            st.session_state.history.clear()
            st.session_state.future.clear()
            st.success("Circuit Bell created.")

        if st.button("🔄 GHZ State", use_container_width=True):
            st.session_state.qc = QuantumCircuit(3)
            st.session_state.qc.h(0)
            st.session_state.qc.cx(0, 1)
            st.session_state.qc.cx(1, 2)
            st.session_state.num_qubits = 3
            st.success("Circuit GHZ State created.")

        if st.button("🌊 Superposition", use_container_width=True):
            st.session_state.qc = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                st.session_state.qc.h(i)
            st.session_state.num_qubits = num_qubits
            st.success("Circuit Superposition loaded!")
    
    # reseteaza circuitul daca s-a schimbat numarul de qubits
    if num_qubits != st.session_state.num_qubits:
        st.session_state.qc = QuantumCircuit(num_qubits)
        st.session_state.num_qubits = num_qubits
        st.session_state.history.clear()
        st.session_state.future.clear()
        st.success("Circuit has been reseted.")


    def save_state():
        st.session_state.history.append(copy.deepcopy(st.session_state.qc))
        st.session_state.future.clear()

    qc = st.session_state.qc

    tab1, tab2, tab3 = st.tabs(["🛠️ Circuit Constructor", "📊 Analysis", "🎯 Measurements"])

    with tab1:
        st.markdown("### 🛠️ Circuit Constructor")

        col_gate, col_params = st.columns([1, 2])

        with col_gate:
            gate_cat = st.selectbox("Gate Category", ["Base", "Rotations", "Control", "Advanced"])

            if gate_cat == "Base":
                gates = ["H", "X", "Y", "Z", "S", "T"]
            elif gate_cat == "Rotations":
                gates = ["RX", "RY", "RZ", "U1", "U2", "U3"]
            elif gate_cat == "Control":
                gates = ["CX", "CY", "CZ", "CCX", "MCX"]
            else:
                gates = ["SWAP", "QFT", "Custom"]

            gate = st.selectbox("Select Gate", gates)

        with col_params:
            qubits = []
            angle = 0

            EXPLICATII_PORTI = {
                "H": "🔄 **Hadamard**: Creates superposition |0⟩ → (|0⟩+|1⟩)/√2",
                "X": "🔁 **Pauli-X**: Quantum bit flip |0⟩ ↔ |1⟩",
                "Y": "🌀 **Pauli-Y**: Rotation around the Y-axis of the Bloch sphere",
                "Z": "⚡ **Pauli-Z**: Phase flip |1⟩ → -|1⟩",
                "S": "📐 **S Gate**: π/2 rotation around the Z-axis",
                "T": "🔺 **T Gate**: π/4 rotation around the Z-axis",
                "RX": "↩️ **X Rotation**: Parameterized rotation around the X-axis",
                "RY": "🔄 **Y Rotation**: Parameterized rotation around the Y-axis", 
                "RZ": "🌀 **Z Rotation**: Parameterized rotation around the Z-axis",
                "CX": "🔗 **CNOT**: X gate conditioned on control qubit state",
                "CCX": "🔗🔗 **Toffoli**: X gate conditioned on two control qubits",
                "SWAP": "↔️ **SWAP**: Swaps the states of two qubits",
                "QFT": "🌊 **QFT**: Quantum Fourier Transform"
            }

            if gate in EXPLICATII_PORTI:
                st.info(EXPLICATII_PORTI[gate])

            if gate in ["H", "X", "Y", "Z", "S", "T"]:
                target = st.multiselect("🎯 Target Qubit(s)", list(range(num_qubits)), default=num_qubits-1)
                qubits = [target]

            # rotation gates
            elif gate in ["RX", "RY", "RZ"]:
                target = st.selectbox("🎯 Target Qubit", list(range(num_qubits)))
                angle = st.slider(f"🔄 Angle of rotation (radians)", 0.0, 2*np.pi, np.pi/2, 0.1)
                qubits = [target, angle]


            elif gate in ["CX", "CY", "CZ"]:
                control = st.selectbox("🔒 Control Qubit", list(range(num_qubits)))
                target = st.selectbox("🎯 Target Qubit", [q for q in range(num_qubits) if q != control])
                qubits = [control, target]
            elif gate == "CCX":
                controls = st.multiselect("🔒 Control Qubit(s)", list(range(num_qubits)))
                if len(controls) == 2:
                    remain = [q for q in range(num_qubits) if q not in controls]
                    target = st.selectbox("🎯 Target Qubit", remain)
                    qubits = [controls[0], controls[1], target]
            elif gate == "MCX":
                controls = st.multiselect("🔒 Control Qubit(s)", list(range(num_qubits)))
                if len(controls) >= 1:
                    remain = [q for q in range(num_qubits) if q not in controls]
                    target = st.selectbox("🎯 Target Qubit", remain)
                    qubits = [controls, target]
            elif gate == "SWAP":
                qubit1 = st.selectbox("🎯 Qubit 1", list(range(num_qubits)))
                qubit2 = st.selectbox("🎯 Qubit 2", [q for q in range(num_qubits) if q != qubit1])
                qubits = [qubit1, qubit2]
            elif gate == "QFT":
                qft_qubits = st.multiselect("🎯 Qubits for QFT", list(range(num_qubits)))
                qubits = [qft_qubits]

        col_apply, col_undo, col_redo, col_reset = st.columns(4)

        with col_apply:
            if st.button("✅ Aplică Poarta", type="primary", use_container_width=True):
                save_state()
                try:
                    if gate == "H" and qubits[0]:
                        for q in qubits[0]: qc.h(q)
                    elif gate == "X" and qubits[0]:
                        for q in qubits[0]: qc.x(q)
                    elif gate == "Y" and qubits[0]:
                        for q in qubits[0]: qc.y(q)
                    elif gate == "Z" and qubits[0]:
                        for q in qubits[0]: qc.z(q)
                    elif gate == "S" and qubits[0]:
                        for q in qubits[0]: qc.s(q)
                    elif gate == "T" and qubits[0]:
                        for q in qubits[0]: qc.t(q)
                    elif gate == "RX":
                        qc.rx(qubits[1], qubits[0])
                    elif gate == "RY":
                        qc.ry(qubits[1], qubits[0])
                    elif gate == "RZ":
                        qc.rz(qubits[1], qubits[0])
                    elif gate == "CX" and len(qubits) == 2:
                        qc.cx(qubits[0], qubits[1])
                    elif gate == "CY" and len(qubits) == 2:
                        qc.cy(qubits[0], qubits[1])
                    elif gate == "CZ" and len(qubits) == 2:
                        qc.cz(qubits[0], qubits[1])
                    elif gate == "CCX" and len(qubits) == 3:
                        qc.ccx(qubits[0], qubits[1], qubits[2])
                    elif gate == "MCX" and len(qubits) == 2:
                        qc.mcx(qubits[0], qubits[1])
                    elif gate == "SWAP" and len(qubits) == 2:
                        qc.swap(qubits[0], qubits[1])
                    elif gate == "QFT" and qubits[0]:
                        from qiskit.circuit.library import QFT
                        qft_gate = QFT(len(qubits[0]))
                        qc.append(qft_gate, qubits[0])
                        
                    st.success(f"✅ Gate {gate} applied!")
                    
                    if st.session_state.auto_measure:
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with col_undo:
            if st.button("↩️ Undo") and st.session_state.history:
                st.session_state.future.append(copy.deepcopy(st.session_state.qc))
                st.session_state.qc = st.session_state.history.pop()
                st.success("Undo applied!")

        with col_redo:
            if st.button("↪️ Redo") and st.session_state.future:
                st.session_state.history.append(copy.deepcopy(st.session_state.qc))
                st.session_state.qc = st.session_state.future.pop()
                st.success("Redo applied!")

        with col_reset:
            if st.button("🗑️ Reset Circuit"):
                st.session_state.qc = QuantumCircuit(num_qubits)
                st.session_state.history.clear()
                st.session_state.future.clear()
                st.success("Circuit has been reset.")

        qc = st.session_state.qc

        if qc.size() > 0:
            try:
                st.markdown("-----")
                st.markdown("### 🖼️ Circuit Visualization")
                img = qc.draw("latex")
                padded_img = ImageOps.expand(img, border=20, fill=(255, 255, 255))
                st.image(padded_img)
            except:
                st.code(str(qc))
    
    with tab2:
        st.markdown("### 📊 Circuit Analysis")

        col_viz, col_info = st.columns([2, 1])

        with col_viz:
            if qc.size() > 0:
                try:
                    img = qc.draw("latex")
                    padded_img = ImageOps.expand(img, border=20, fill=(255, 255, 255))
                    st.image(padded_img)
                except:
                    st.code(str(qc))
            else:
                st.info("🔄 Circuit is empty. Add gates to visualize.")

        
        with col_info:
            st.markdown("#### 📋 Circuit Information")

            depth = qc.depth()
            width = qc.width()
            size = qc.size()

            st.metric("🔍 Circuit Depth", depth)
            st.metric("📏 Circuit Width", width)
            st.metric("📐 Circuit Size", size)

            if size > 0:
                gate_counts = {}
                for instruction in qc.data:
                    gate_name = instruction[0].name
                    gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

                st.markdown("#### Gate Distribution")
                for gate_name, count in gate_counts.items():
                    st.write(f"**{gate_name}**: {count} times")

        if st.session_state.show_statevector and qc.size() > 0:
            st.markdown("### 📊 Statevector Analysis")

            try:
                psi = Statevector.from_instruction(qc)

                col_bloch, col_table = st.columns([2, 1])

                with col_bloch:
                    if num_qubits <= 3:
                        if st.button("🔮 Show Bloch Sphere"):
                            fig = plot_bloch_multivector(psi, figsize=(4 * min(num_qubits, 3), 4))
                            st.pyplot(fig)
                    else:
                        st.info("Bloch sphere visualization is limited to 3 qubits or fewer.")

                with col_table:
                    st.markdown("#### Amplitude Table")

                    amplitudes = psi.data
                    probs = psi.probabilities()

                    data = []
                    for i, (amp, prob) in enumerate(zip(amplitudes, probs)):
                        if abs(prob) > 1e-10:
                            bin_state = format(i, f'0{num_qubits}b')
                            data.append({
                                'Stare': f"|{bin_state}⟩",
                                'Probabiltity': f"{prob:.4f}",
                                'Amplitude': f"{amp:.3f}"
                            })
                    
                    if data:
                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)

                        entropy = -sum(prob * np.log2(prob) for prob in probs if prob > 0)
                        st.metric("🔍 Entropy", f"{entropy:.4f}")

            except Exception as e: 
                st.error(f"❌ Eroare la calcularea statevector: {e}")

            except Exception as e:
                st.error(f"❌ Eroare la calcularea statevector: {e}")

    with tab3:
        st.markdown("### 🎯 Measurements")

        if qc.size() == 0:
            st.info("🔄 Circuit is empty. Add gates to measure.")
        else:
            col_measure, col_results = st.columns([1, 2])

            with col_measure:
                st.markdown("#### 📏 Measurement Settings")

                measure_type = st.radio("Measurement Type", ["All Qubits", "Selective", "Partial"])

                measured_qubits = []

                if measure_type == "All Qubits":
                    measured_qubits = list(range(num_qubits))
                elif measure_type == "Selective":
                    measured_qubits = st.multiselect("Select Qubits to Measure", list(range(num_qubits)), default=list(range(num_qubits)))
                elif measure_type == "Partial":
                    n_measure = st.slider("Number of Qubits to Measure", 1, num_qubits, 1)
                    measured_qubits = list(range(n_measure))

                if st.button("Run Measurements", type="primary"):
                    qc_sim = qc.copy() # make a copy to avoid modifying the original circuit

                    if measured_qubits:
                        qc_sim.add_register(ClassicalRegister(len(measured_qubits)))
                        for i, qubit in enumerate(measured_qubits):
                            qc_sim.measure(qubit, i)
                    else:
                        qc_sim.measure_all()

                    backend = AerSimulator()
                    transpiled = transpile(qc_sim, backend)
                    job = backend.run(transpiled, shots=shots)
                    results = job.result()
                    counts = results.get_counts()

                    st.session_state.last_results = {
                        'counts': counts,
                        'measured_qubits': measured_qubits,
                        'shots': shots
                    }

            with col_results:
                if 'last_results' in st.session_state:
                    st.markdown("#### 📈 Measurement Results")
                    
                    counts = st.session_state.last_results['counts']
                    measured_qubits = st.session_state.last_results['measured_qubits']
                    total_shots = st.session_state.last_results['shots']
                    
                    # Histogram
                    fig = plot_histogram(counts, figsize=(12, 6), 
                                       title=f"Measurement results ({total_shots} shots)")
                    st.pyplot(fig)
                    
                    # Statistics
                    most_common = max(counts.items(), key=lambda x: x[1])
                    least_common = min(counts.items(), key=lambda x: x[1])
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("🎯 Most Common", 
                                f"{most_common[0]}", 
                                f"{most_common[1]}/{total_shots}")
                    with col_stat2:
                        st.metric("📊 Unique states", len(counts))
                    with col_stat3:
                        variance = np.var(list(counts.values()))
                        st.metric("📈 Variation", f"{variance:.1f}")
                    
                    st.markdown("#### 📋 Detailed Results")
                    results_data = []
                    for state, count in sorted(counts.items(), key=lambda x: -x[1]):
                        probability = count / total_shots
                        results_data.append({
                            'Status': state,
                            'Frequency': count,
                            'Probability': f"{probability:.4f}",
                            'Percentage': f"{probability*100:.2f}%"
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(df_results, use_container_width=True)
                
    if qc.size() > 0:
        st.markdown("---")
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        
        with col_info1:
            st.metric("🎛️ Total Gates", qc.size())
        with col_info2:
            st.metric("📏 Depth", qc.depth())  
        with col_info3:
            try:
                psi = Statevector.from_instruction(qc)
                entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in psi.probabilities())
                st.metric("🔀 Entropy", f"{entropy:.2f}")
            except:
                st.metric("🔀 Entropy", "N/A")
        with col_info4:
            complexity = qc.size() * qc.depth()
            st.metric("⚡ Complexity", complexity)

elif page == "🚄 Performance & Optimization Lab":
    st.markdown("<h2 class='algo-header'>🚄 Crypto Performance Reality Check</h2>", unsafe_allow_html=True)

    if 'crypto_performance' not in st.session_state:
        class CryptoPerformanceAnalyzer:
            def password_crack_simulation(self, password_length: int, method: str):
                print(f"🔓 Simulating password crack: {password_length} chars, method: {method}")

                total_combinations = 95 ** password_length

                if method == "Single CPU":
                    ops_per_second = 1_000_000
                    time_seconds = total_combinations / (2 * ops_per_second)

                elif method == "GPU Cluster":
                    ops_per_second = 100_000_000_000
                    time_seconds = total_combinations / (2 * ops_per_second)

                elif method == "Quantum (Grover)":
                    quantum_iterations = math.sqrt(total_combinations)
                    quantum_ops_per_second = 10_000_000
                    time_seconds = quantum_iterations / quantum_ops_per_second

                else:
                    time_seconds = float('inf')

                if time_seconds == float('inf'):
                    time_str = "Not feasible"
                elif time_seconds < 60:
                    time_str = f"{time_seconds:.2f} seconds"
                elif time_seconds < 3600:
                    time_str = f"{time_seconds / 60:.1f} minutes"
                elif time_seconds < 86400:
                    time_str = f"{time_seconds / 3600:.1f} hours"
                elif time_seconds < 31536000:
                    time_str = f"{time_seconds / 86400:.1f} days"
                else:
                    years = time_seconds / 31536000
                    if years > 1e6:
                        time_str = f"{years:.2e} years"
                    else:
                        time_str = f"{years:.0f} years"

                return {
                    'method': method,
                    'password_length': password_length,
                    'combinations': total_combinations,
                    'time_seconds': time_seconds,
                    'time_readable': time_str,
                    'practical': time_seconds < 86400 * 365
                }

            def crypto_algorithm_benchmark(self, algorithm: str, key_size: int):
                performance_data = {
                    'AES-128': {
                        'encrypt': 1_000_000,
                        'crack_classical': 2 ** 127,
                        'crack_quantum_grover': 2 ** 63,  
                        'shor_vulnerable': False
                    },
                    'AES-256': {
                        'encrypt': 800_000,
                        'crack_classical': 2 ** 255,
                        'crack_quantum_grover': 2 ** 127,
                        'shor_vulnerable': False
                    },
                    'RSA-1024': {
                        'encrypt': 50_000,
                        'crack_classical': 10 ** 15,
                        'crack_quantum_grover': 10 ** 15, 
                        'shor_vulnerable': True, 
                        'shor_time_ideal': 'Minutes',
                        'shor_time_realistic': 'Hours to days'
                    },
                    'RSA-2048': {
                        'encrypt': 12_000,
                        'crack_classical': 10 ** 25,
                        'crack_quantum_grover': 10 ** 25,
                        'shor_vulnerable': True,
                        'shor_time_ideal': 'Minutes',
                        'shor_time_realistic': 'Hours to days'
                    },
                    'RSA-4096': {
                        'encrypt': 3_000,
                        'crack_classical': 10 ** 40,
                        'crack_quantum_grover': 10 ** 40,
                        'shor_vulnerable': True,
                        'shor_time_ideal': 'Hours',
                        'shor_time_realistic': 'Days to weeks'
                    },
                    'Kyber-512': {
                        'encrypt': 100_000,
                        'crack_classical': 2 ** 128,
                        'crack_quantum_grover': 2 ** 64, 
                        'shor_vulnerable': False,
                        'quantum_resistant': True
                    },
                    'Kyber-1024': {
                        'encrypt': 50_000,
                        'crack_classical': 2 ** 256,
                        'crack_quantum_grover': 2 ** 128,
                        'shor_vulnerable': False,
                        'quantum_resistant': True
                    }
                }

                if algorithm not in performance_data:
                    return None

                data = performance_data[algorithm]

                classical_years = data['crack_classical'] / (1e12 * 31536000)

                if data.get('shor_vulnerable', False):
                    quantum_time = data.get('shor_time_realistic', 'Hours to days')
                    quantum_years = 0.001
                    currently_broken = True
                else:
                    quantum_years = data['crack_quantum_grover'] / (1e6 * 31536000)
                    quantum_time = f"{quantum_years:.2e} years" if quantum_years > 1 else "< 1 year"
                    currently_broken = False

                return {
                    'algorithm': algorithm,
                    'encrypt_ops_per_sec': data['encrypt'],
                    'classical_crack_years': classical_years,
                    'quantum_crack_time': quantum_time,
                    'quantum_resistant': data.get('quantum_resistant', False),
                    'currently_broken': currently_broken,
                    'shor_vulnerable': data.get('shor_vulnerable', False)
                }


        st.session_state.crypto_performance = CryptoPerformanceAnalyzer()

    tab1, tab2 = st.tabs(["🔓 Password Security", "⚖️ Algorithm Performance"])

    with tab1:
        st.markdown("### 🔓 How Long to Crack Your Password?")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
                <div class="modern-card">
                    <h4>Password Cracking Reality</h4>
                    <p>See how long it takes to crack passwords with different methods. Note: Current quantum computers are NOT better than classical computers for password cracking!</p>
                </div>
                """, unsafe_allow_html=True)

            password_length = st.slider("Password length (characters)", 4, 16, 8)

            if st.button("🔍 Calculate Crack Times", type="primary"):
                methods = ["Single CPU", "GPU Cluster", "Quantum (Grover)"]
                results = []

                for method in methods:
                    result = st.session_state.crypto_performance.password_crack_simulation(password_length, method)
                    results.append(result)

                st.markdown(f"#### ⏱️ Time to crack {password_length}-character password:")

                for result in results:
                    if result['method'] == "Quantum (Grover)":
                        color_class = "security-safe"  # Show quantum as advantageous
                        practical_text = "⚛️ Quantum advantage demonstrated!"
                    else:
                        color_class = "security-safe" if not result['practical'] else "attack-result"
                        practical_text = "✅ Secure" if not result['practical'] else "❌ Vulnerable"

                    st.markdown(f"""
                        <div class="{color_class}">
                            <h4>{result['method']}: {result['time_readable']}</h4>
                            <p>{practical_text}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Add explanation
                st.markdown("""
                    <div class="advanced-box">
                        <h4>🧠 Understanding the Results:</h4>
                        <ul>
                            <li><strong>Single CPU:</strong> Standard computer with secure password hashing</li>
                            <li><strong>GPU Cluster:</strong> Specialized hardware for password cracking</li>
                            <li><strong>Quantum (Grover):</strong> Ideal quantum computer with Grover's algorithm - √N speedup</li>
                        </ul>
                        <p><strong>Quantum Advantage:</strong> Grover's algorithm provides quadratic speedup, making quantum computers significantly faster for brute force attacks when fault-tolerant quantum computers become available.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Filter out infinite values for plotting
                finite_results = [r for r in results if r['time_seconds'] != float('inf')]
                if finite_results:
                    times_seconds = [r['time_seconds'] for r in finite_results]
                    methods_finite = [r['method'] for r in finite_results]
                    times_log = [math.log10(max(t, 1)) for t in times_seconds]

                    fig_crack = go.Figure()
                    fig_crack.add_trace(go.Bar(
                        x=methods_finite,
                        y=times_log,
                        text=[r['time_readable'] for r in finite_results],
                        textposition='auto',
                        marker_color=['orange', 'red', 'blue', 'purple'][:len(finite_results)]
                    ))

                    fig_crack.update_layout(
                        title="Password Cracking Time Comparison (Log Scale)",
                        xaxis_title="Attack Method",
                        yaxis_title="Time (log10 seconds)",
                        template="plotly_white"
                    )

                    st.plotly_chart(fig_crack, use_container_width=True)

        with col2:
            st.markdown("### 💡 Password Security Recommendations")

            if password_length < 8:
                recommendation = "🔴 DANGEROUS"
                advice = "Increase to at least 12 characters immediately"
            elif password_length < 12:
                recommendation = "🟡 VULNERABLE to future quantum"
                advice = "Use 16+ characters to resist quantum attacks"
            elif password_length < 16:
                recommendation = "🟠 MODERATE"
                advice = "Good for now, but consider longer passwords"
            else:
                recommendation = "🟢 STRONG"
                advice = "Excellent security against current threats"

            st.markdown(f"""
                <div class="performance-metric">
                    <h4>Security Level: {recommendation}</h4>
                    <p><strong>Recommendation:</strong> {advice}</p>
                    <p><strong>Key insight:</strong> Quantum computers will be much faster at password cracking using Grover's algorithm</p>
                </div>
                """, unsafe_allow_html=True)

            # st.markdown("#### 📉 Password Security Over Time")

            years = np.array([2024, 2026, 2028, 2030, 2035])
            strength_8_char = np.array([60, 30, 15, 5, 0])
            strength_12_char = np.array([90, 70, 50, 30, 15])
            strength_16_char = np.array([98, 90, 80, 70, 50])

            fig_degradation = go.Figure()
            fig_degradation.add_trace(
                go.Scatter(x=years, y=strength_8_char, name="8 characters", line=dict(color='red', width=3)))
            fig_degradation.add_trace(
                go.Scatter(x=years, y=strength_12_char, name="12 characters", line=dict(color='orange', width=3)))
            fig_degradation.add_trace(
                go.Scatter(x=years, y=strength_16_char, name="16 characters", line=dict(color='green', width=3)))

            # fig_degradation.update_layout(
            #     title="Password Security Over Time (Including Quantum Threat)",
            #     xaxis_title="Year",
            #     yaxis_title="Security Level (%)",
            #     template="plotly_white"
            # )

            fig_degradation.update_layout(
                title={
                    'text': "📉 Password Security Over Time",
                    'font': {'size': 20, 'color': 'white'}
                },
                xaxis_title="Year",
                yaxis_title="Security Level (%)",
                yaxis_type="log",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                hovermode="x unified"
            )

            st.plotly_chart(fig_degradation, use_container_width=True)

    with tab2:
        st.markdown("### ⚖️ Cryptographic Algorithm Performance")

        col1, col2 = st.columns([1, 1])

        with col1:
            algorithm = st.selectbox("Choose Algorithm", [
                "AES-128", "AES-256", "RSA-1024", "RSA-2048", "RSA-4096", "Kyber-512", "Kyber-1024"
            ])

            if st.button("📊 Analyze Performance", type="primary"):
                result = st.session_state.crypto_performance.crypto_algorithm_benchmark(algorithm, 0)

                if result:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🚀 Speed", f"{result['encrypt_ops_per_sec']:,} ops/sec")
                    with col_b:
                        st.metric("🛡️ Classical Security", f"{result['classical_crack_years']:.2e} years")
                    with col_c:
                        quantum_secure = "✅ Secure" if result['quantum_resistant'] else (
                            "❌ Broken by Shor" if result['shor_vulnerable'] else "⚠️ Reduced")
                        st.metric("⚛️ Quantum Status", quantum_secure)

                    # Security status with proper explanation
                    if result['shor_vulnerable']:
                        st.markdown(f"""
                            <div class="attack-result">
                                <h4>🚨 CRITICAL: Vulnerable to Shor's Algorithm!</h4>
                                <p><strong>{algorithm}</strong> can be broken by quantum computers using Shor's algorithm in {result['quantum_crack_time']}</p>
                                <p><strong>Why:</strong> Shor's algorithm efficiently factors large numbers, breaking RSA's security foundation</p>
                                <p><strong>Action needed:</strong> Migrate to post-quantum algorithms (like Kyber) immediately</p>
                            </div>
                            """, unsafe_allow_html=True)
                    elif result['quantum_resistant']:
                        st.markdown(f"""
                            <div class="security-safe">
                                <h4>✅ Quantum Resistant</h4>
                                <p><strong>{algorithm}</strong> remains secure against known quantum algorithms</p>
                                <p><strong>Quantum crack time:</strong> {result['quantum_crack_time']}</p>
                                <p><strong>Why:</strong> Based on mathematical problems that quantum computers can't solve efficiently</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="performance-metric">
                                <h4>⚠️ Quantum Weakened (but not broken)</h4>
                                <p><strong>{algorithm}</strong> has reduced security against quantum computers using Grover's algorithm</p>
                                <p><strong>Quantum crack time:</strong> {result['quantum_crack_time']}</p>
                                <p><strong>Why:</strong> Grover's algorithm provides quadratic speedup for brute force attacks</p>
                            </div>
                            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 Algorithm Comparison")

            algorithms = ["AES-128", "AES-256", "RSA-2048", "Kyber-512"]
            speeds = [1000000, 800000, 12000, 100000]
            quantum_status = ['Weakened', 'Weakened', 'Broken', 'Secure']

            colors = {
                'Broken': 'red',
                'Weakened': 'orange',
                'Secure': 'green'
            }
            bar_colors = [colors[status] for status in quantum_status]

            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                x=algorithms,
                y=speeds,
                marker_color=bar_colors,
                text=[f"{s:,}" for s in speeds],
                textposition='auto'
            ))

            fig_comparison.update_layout(
                title="Algorithm Performance vs Quantum Security",
                xaxis_title="Algorithm",
                yaxis_title="Operations per Second",
                template="plotly_white"
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

            st.markdown("""
                <div class="advanced-box">
                    <h4>🎯 Key Insights:</h4>
                    <ul>
                        <li><strong>RSA (Red):</strong> Completely broken by Shor's algorithm - migrate urgently</li>
                        <li><strong>AES (Orange):</strong> Weakened by Grover but still practical security</li>
                        <li><strong>Kyber (Green):</strong> Quantum-resistant with good performance</li>
                        <li><strong>Trade-offs:</strong> Post-quantum algorithms balance security vs performance</li>
                    </ul>
                    <p><strong>Bottom Line:</strong> The quantum threat is real for RSA, theoretical for passwords, and manageable for symmetric encryption.</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🧠 Grover vs Kyber (simulare educațională)":
    st.markdown("<h2 class='algo-header'>🧠 Grover contra Kyber – Simulare educațională</h2>", unsafe_allow_html=True)

    st.markdown("""
        <div class="box">
          <p>Algoritmul <strong>Grover</strong> accelerează căutarea exhaustivă (~√N pași) și amenință
             criptarea simetrică (ex.: AES). Însă pentru scheme post-quantum precum
             <strong>Kyber</strong> (un mecanism de încapsulare a cheii, KEM) acest atac devine
             nepractic.</p>
        </div>
        """, unsafe_allow_html=True)

    mod = st.radio(
        "🔧 Alege scenariul de simulare:",
        ["🔬 Demo rapid (până la 12 biți)", "🛡️ Scenariu realist Kyber-512 (256 biți)"],
    )

    if mod.startswith("🔬"):
        n = st.slider("🔐 Număr biți cheie (demo)", 2, 12, 4)
    else:
        n = 256

    # Estimare Grover
    iterations = int((math.pi / 4) * math.sqrt(2 ** n))
    st.markdown(f"""
        <div class="metric_box">
          <h4>ℹ️ Estimare Grover:</h4>
          <ul>
            <li>Qubiți logici necesari: <strong>{n}</strong></li>
            <li>Iterații Grover: <strong>{iterations:,}</strong></li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    if n <= 12:
        st.markdown("### 🧪 Circuit demonstrativ Grover (mică dimensiune)")
        from qiskit import QuantumCircuit
        from qiskit.visualization import plot_histogram

        grover_qc = QuantumCircuit(n)
        grover_qc.h(range(n))
        for _ in range(iterations):
            grover_qc.h(range(n))
            grover_qc.x(range(n))
            grover_qc.h(n - 1)
            grover_qc.mcx(list(range(n - 1)), n - 1)
            grover_qc.h(n - 1)
            grover_qc.x(range(n))
            grover_qc.h(range(n))
        grover_qc.measure_all()

        st.pyplot(grover_qc.draw(output='mpl', fold=-1))

    if n >= 128:
        st.markdown("""
            <div class="box">
                <h3>🚫 Atac Grover imposibil în practică pentru n ≥ 256</h3>
                <p>Un atac Grover ar necesita:</p>
                <ul>
                    <li>~2<sup>128</sup> iterații pentru cheia Kyber-512</li>
                    <li>Milione de qubiți fizici stabili</li>
                    <li>Milenii de rulări fără erori</li>
                </ul>
                <p><strong>➡️ Concluzie:</strong> Kyber este practic imposibil de spart cu Grover. Este considerat <em>post-quantum secure</em>.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📈 Cum cresc iterațiile Grover vs. dimensiunea cheii")
    n_vals = np.arange(2, (12 if mod.startswith('🔬') else 257))
    iters = (np.pi / 4 * np.sqrt(2 ** n_vals)).astype(int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=n_vals, y=iters,
            mode="lines+markers",
            name="Iterații Grover",
            line=dict(width=3),
            hovertemplate="n=%{x} → %{y} iterații<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Dimensiunea cheii (biți)",
        yaxis_title="Iterații (log-scale)",
        yaxis_type="log",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
        <div class="safe-box">
          <h4>🔑 De reținut</h4>
          <ul>
            <li><strong>Grover</strong> reduce căutarea de la <em>O(2ⁿ)</em> la <em>O(2ⁿ⁄²)</em>.</li>
            <li>Pentru Kyber-512 (256 biți) înseamnă tot ~2<sup>128</sup> pași – imposibil în practică.</li>
            <li>Standardele NIST post-quantum (Kyber, Dilithium…) rămân sigure în fața acestui atac.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔥 Quantum Threat Dashboard":


    st.markdown("""
    <div class="modern-card">
        <h2>🔥 Real-Time Quantum Threat Assessment</h2>
        <p>Live analysis of your security posture against quantum attacks</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🛡️ Personal Security Test")
        
        st.markdown("#### Password Vulnerability Test")
        password = st.text_input("Enter a password to test:", type="password", 
                                help="We don't store or transmit your password - all calculations are local")
        
        if password:
            password_length = len(password)
            
            has_lower = any(c.islower() for c in password)
            has_upper = any(c.isupper() for c in password)
            has_digits = any(c.isdigit() for c in password)
            has_special = any(not c.isalnum() for c in password)
            
            charset_size = 0
            if has_lower: charset_size += 26
            if has_upper: charset_size += 26  
            if has_digits: charset_size += 10
            if has_special: charset_size += 20
            
            total_combinations = charset_size ** password_length
            
            classical_ops_per_sec = 100_000_000_000  # gpu cluster speed
            classical_seconds = total_combinations / (2 * classical_ops_per_sec)
            
            quantum_iterations = math.sqrt(total_combinations)
            quantum_ops_per_sec = 1_000_000 
            quantum_seconds = quantum_iterations / quantum_ops_per_sec
            
            def seconds_to_readable(seconds):
                if seconds < 60:
                    return f"{seconds:.1f} seconds"
                elif seconds < 3600:
                    return f"{seconds/60:.1f} minutes"
                elif seconds < 86400:
                    return f"{seconds/3600:.1f} hours"
                elif seconds < 31536000:
                    return f"{seconds/86400:.1f} days"
                else:
                    years = seconds / 31536000
                    if years > 1e12:
                        return f"{years:.1e} years"
                    else:
                        return f"{years:,.0f} years"
            
            classical_time = seconds_to_readable(classical_seconds)
            quantum_time = seconds_to_readable(quantum_seconds)
            
            if quantum_seconds < 86400: 
                threat_level = "🔴 CRITICAL"
                threat_class = "attack-result"
                advice = "Change immediately! Use 16+ characters with mixed case, numbers, symbols"
            elif quantum_seconds < 31536000: 
                threat_level = "🟠 HIGH RISK"
                threat_class = "performance-metric"
                advice = "Vulnerable to quantum attacks. Consider longer password"
            elif quantum_seconds < 31536000 * 10: 
                threat_level = "🟡 MODERATE"
                threat_class = "performance-metric" 
                advice = "Good for now, but quantum computers are getting stronger"
            else:
                threat_level = "🟢 SECURE"
                threat_class = "security-safe"
                advice = "Strong against quantum attacks for foreseeable future"
            
            st.markdown(f"""
            <div class="{threat_class}">
                <h4>{threat_level}</h4>
                <p><strong>Classical crack time:</strong> {classical_time}</p>
                <p><strong>Quantum crack time:</strong> {quantum_time}</p>
                <p><strong>Recommendation:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Your Systems Check")
        systems = {
            "RSA-2048 (SSH, TLS)": st.checkbox("RSA-2048 (SSH, TLS)", value=True),
            "ECC-256 (Mobile, IoT)": st.checkbox("ECC-256 (Mobile, IoT)", value=True),
            "AES-128 (File encryption)": st.checkbox("AES-128 (File encryption)", value=True),
        }
        
        vulnerable_systems = []
        for system, enabled in systems.items():
            if enabled:
                if "RSA" in system or "ECC" in system:
                    vulnerable_systems.append(system)
        
        if vulnerable_systems:
            st.markdown("#### 🚨 Quantum-Vulnerable Systems:")
            for system in vulnerable_systems:
                st.error(f"❌ {system} - **WILL BE BROKEN** by quantum computers")
        
        if systems["AES-128 (File encryption)"]:
            st.warning("⚠️ AES-128 - Security reduced to 64-bit (weakened but not broken)")
    
    with col2:
        st.markdown("### 📊 Global Quantum Progress")
        
        milestones = {
            "Google Supremacy": {"year": 2019, "qubits": 53, "status": "✅"},
            "IBM Eagle": {"year": 2021, "qubits": 127, "status": "✅"},
            "IBM Condor": {"year": 2023, "qubits": 1121, "status": "✅"},
            "Logical Qubits": {"year": 2024, "qubits": 5000, "status": "🟡"},
            "Fault-Tolerant": {"year": 2028, "qubits": 50000, "status": "🔮"},
            "Crypto-Breaking": {"year": 2032, "qubits": 1000000, "status": "🎯"}
        }
        
        st.markdown("#### Quantum Computing Timeline")
        for name, data in milestones.items():
            status_color = "#4caf50" if data["status"] == "✅" else "#ffa726" if data["status"] == "🟡" else "#667eea"
            
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, {status_color}20, transparent); 
                        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
                        border-left: 4px solid {status_color};">
                <strong>{data['year']}</strong> - {name} {data['status']}
                <br><small>{data['qubits']:,} qubits</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### 🎯 When Will Your Crypto Break?")
        
        crypto_break_times = {
            "RSA-1024": 2028,
            "RSA-2048": 2030, 
            "RSA-4096": 2032,
            "ECC-256": 2029,
            "ECC-384": 2031,
            "Bitcoin signatures": 2030
        }
        
        fig_timeline = go.Figure()
        
        algos = list(crypto_break_times.keys())
        break_years = list(crypto_break_times.values())
        colors = ['red' if year <= 2030 else 'orange' for year in break_years]
        
        fig_timeline.add_trace(go.Bar(
            x=algos,
            y=break_years,
            marker_color=colors,
            text=[f"{year}" for year in break_years],
            textposition='auto'
        ))
        
        fig_timeline.update_layout(
            title="🎯 Crypto Break Timeline",
            yaxis_title="Predicted Break Year",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            yaxis=dict(range=[2025, 2035])
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("#### ⚡ Current Quantum Computing Power")
        
        import datetime
        days_since_2024 = (datetime.datetime.now() - datetime.datetime(2024, 1, 1)).days
        simulated_qubits = 1000 + (days_since_2024 * 2.7) 
        
        progress_to_crypto_breaking = (simulated_qubits / 1000000) * 100  
        
        st.metric("Current Estimated Logical Qubits", f"{simulated_qubits:,.0f}")
        st.metric("Progress to Crypto-Breaking Computer", f"{progress_to_crypto_breaking:.2f}%")
        
        st.progress(min(progress_to_crypto_breaking / 100, 1.0))
        
        if progress_to_crypto_breaking < 1:
            st.info("🔬 Still in early research phase")
        elif progress_to_crypto_breaking < 10:
            st.warning("⚠️ Significant progress being made")
        else:
            st.error("🚨 Quantum threat becoming real!")
    
    # Action plan section
    st.markdown("---")
    st.markdown("### 🛡️ Your Quantum-Safe Action Plan")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        <div class="security-safe">
            <h4>🔒 Immediate Actions</h4>
            <ul>
                <li>Use 16+ character passwords</li>
                <li>Enable 2FA everywhere</li>
                <li>Inventory current crypto systems</li>
                <li>Start planning PQC migration</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div class="performance-metric">
            <h4>📅 2025-2027 Timeline</h4>
            <ul>
                <li>Pilot post-quantum algorithms</li>
                <li>Hybrid classical-quantum crypto</li>
                <li>Update critical systems first</li>
                <li>Train security teams</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown("""
        <div class="advanced-box">
            <h4>🔮 2028+ Future</h4>
            <ul>
                <li>Full post-quantum deployment</li>
                <li>Quantum-safe certificates</li>
                <li>Legacy system isolation</li>
                <li>Continuous quantum monitoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea, #764ba2); 
            padding: 2rem; border-radius: 20px; text-align: center; 
            margin: 3rem 0; color: white;">
    <h3>🚀 Ready for the Quantum Future?</h3>
    <p style="font-size: 1.1rem; opacity: 0.9;">
        Don't wait for quantum computers to break your security. Start preparing today.
    </p>
</div>
""", unsafe_allow_html=True)

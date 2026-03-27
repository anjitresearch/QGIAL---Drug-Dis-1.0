# QGIAL - Quantum Generative Intelligence and Adaptive Learning
## Deploy on Render

This repository contains the QGIAL web dashboard for quantum-enhanced drug discovery.

### 🚀 Quick Deploy on Render

1. **Fork or Clone** this repository:
   ```bash
   git clone https://github.com/anjitresearch/QGIAL---Drug-Dis-1.0.git
   cd QGIAL---Drug-Dis-1.0
   ```

2. **Push to GitHub** (if forked)

3. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn web_dashboard.app_standalone:app --bind 0.0.0.0:$PORT --workers 1`
     - **Runtime**: `Python 3.11.7`
     - **Root Directory**: `.`

4. **Deploy** - Click "Create Web Service"

### 🌐 Features

- **Interactive Web Dashboard**: Real-time QGIAL pipeline monitoring
- **Quantum Molecular Simulation**: 27-qubit VQC demonstrations
- **Multi-Target Support**: KRAS G12D, PD-L1, SARS-CoV-2 Mpro
- **Live Visualizations**: Fitness progression and optimization metrics
- **Molecular Results**: Top generated candidates with detailed properties
- **Professional Interface**: Modern, responsive design

### 📊 Dashboard Capabilities

- Initialize and run QGIAL pipeline
- Select from multiple drug targets
- Monitor real-time optimization progress
- View top molecular candidates
- Interactive charts and visualizations
- Mobile-responsive design

### 🎯 Demo Data

The dashboard includes realistic simulation data for:
- Molecular optimization trajectories
- Binding affinity improvements
- ADMET property predictions
- PAINS pattern detection
- Multi-objective optimization metrics

### 🧬 Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Visualizations**: Plotly.js
- **Deployment**: Render (Free Tier)
- **Quantum**: Qiskit simulation
- **Chemistry**: RDKit molecular modeling

### 📞 Support

Developed by [Prof. Anjit Raja R](https://www.linkedin.com/in/profanjitraja/) - 2026

---

**Deploy your QGIAL dashboard today and showcase quantum-enhanced drug discovery!** 🚀

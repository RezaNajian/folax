# FOLAX Interactive Microstructure Simulation

This is a Streamlit-based app for generating and simulating 2D/3D microstructures using Voronoi, Fourier, or uploaded images. You can run finite element simulations or the on the fly implicit FOL solver.

## Features

- Generate 2D microstructures:
  - Voronoi
  - Periodic Voronoi
  - Fourier fields
- Generate 3D Fourier microstructures
- Upload your own microstructure images
- Run OTF iFOL solver and compare with FE solver
- Download results in ZIP files

## Requirements

Python 3.11+ recommended. Install dependencies:

```bash
pip install -r requirements.txt


streamlit>=1.30
numpy>=1.25
matplotlib>=3.7
scipy>=1.12
jax>=0.5
plotly>=5.20
Pillow>=10.0
scikit-learn>=1.3


#Run
cd gui
streamlit run folax-streamlit.py

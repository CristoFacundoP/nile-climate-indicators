# nile-climate-indicators
This repository supports the analysis of climate change and potential impacts across the Nile Basin.

The project includes:

Data processing: Downloading, screening, and bias-correcting CMIP6 GCMs.

Indicator computation: Calculating IPCC-recommended climate impact-driver indicators (heat, wet, dry).

Model assessment: Evaluating GCM skill and ranking using statistical tests.

Trend analysis: Detecting mid- and late-century climate trends.

Visualization & tools: Scripts and notebooks for bivariate plots, spatial analysis, and hotspot detection.

The repo is organized into folders for data, scripts, results, docs, and notebooks.
All workflows are designed for reproducibility on both local machines and the Alliance Fir cluster.

## 1. Clone the Repository
# On your local machine or Fir cluster
git clone https://github.com/CristoFacundoP/nile-climate-indicators.git
cd nile-climate-indicators

## 2. Set Up the Environment
module load python/3.11  # on Fir cluster
conda env create -f environment.yml
conda activate nile-climate

## 3. Folder Structure
nile-climate-indicators/
├── data/          # placeholder only (large data stays in /project/ on cluster or workstation)
├── scripts/       # analysis scripts
├── notebooks/     # exploratory notebooks
├── results/       # plots, summary CSVs
├── docs/          # project notes & paper outline
└── environment.yml

# Imperial College London ATES Assessment Tool

An open-source probabilistic assessment tool for Aquifer Thermal Energy Storage (ATES) and Groundwater Heating & Cooling (GHWC) systems.

---

## Features

- Deterministic and probabilistic ATES system performance analysis
- Monte Carlo simulation with multiple distribution types
- Statistical outputs: mean, standard deviation, percentiles
- Sensitivity analysis with parameter importance ranking
- Case saving in three modes: Parameters Only, Parameters + Results, Full State
- Interactive interface built with Streamlit

---


## Live Deployment

Our tool has been successfully deployed to the cloud based on Streamlit Cloud. You can access it directly via:

**Live URL: [https://imperial-ates-assessment-tool.streamlit.app]**

---

## Data Security Notice

**If your data contains sensitive or confidential information, we strongly recommend local deployment.**

For more information on handling sensitive data, please refer to the [Streamlit Secrets Management Documentation](https://docs.streamlit.io/deploy/concepts/secrets).

---

## Local Deployment

### Step 0: Install Python (if not already installed)

This tool requires **Python 3.8 or higher**.

If you do not already have Python installed, please download it from the official website:
- [Python Downloads](https://www.python.org/downloads/)

During installation, make sure to check the option **"Add Python to PATH"** (on Windows).

For Mac users, Python 3 is often preinstalled. You can check by running in terminal:
```bash
python3 --version
```

If you are completely new to Python, please follow this official guide to learn how to run Python programs: [Python Tutorial](https://docs.python.org/3/tutorial/)

### Step 1: Set up a working environment (recommended)

It is strongly recommended to use a virtual environment to avoid dependency conflicts.

Create and activate a virtual environment:
```bash
python -m venv .venv
# On Mac/Linux
source .venv/bin/activate
# On Windows
.venv\Scripts\activate
```

### Step 2: Install the ATES tool

#### Option 1: Clone from GitHub
```bash
git clone https://github.com/esemsc-yy2923/ates-assessment-tool.git
cd ates-assessment-tool
pip install -e .
```

#### Option 2: Install directly from GitHub
```bash
pip install "git+https://github.com/esemsc-yy2923/ates-assessment-tool.git"
```

---

## Usage

After completing installation, you can run the tool from your terminal:

**Run the main assessment tool (Streamlit app):**
```bash
ates-tool
```

By default, the Streamlit app will open in your browser at http://localhost:8501/.

If the command is not found, make sure the installation step was successful.

Alternatively, you can run directly with Python:
```bash
python -m tool.launch
```

---

## Saving Cases

The tool supports three case formats:

**Parameters Only**
- Includes ATES parameters and probability distributions only.
- Use cases: parameter backup, sharing, template creation.

**Parameters + Results**
- Adds deterministic results.
- Use cases: validation, documentation, scenario comparison.

**Full State (Report)**
- Adds complete statistical analysis and sensitivity results.
- Use cases: comprehensive reports, risk assessment.

---

## Development

Contributions are welcome. Please note that the `main` branch is protected and all changes require review and approval before merging.

### Workflow

1. Make sure you are working from the latest `dev` branch:
   ```bash
   git checkout dev
   git pull origin dev
   ```

2. Create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. Implement your changes and commit:
   ```bash
   git add .
   git commit -m "Describe your changes"
   git push origin feature/your-feature-name
   ```

4. Open a Pull Request (PR) from your feature branch into main.

### Guidelines

- Do not commit directly to main.
- Use dev for ongoing development.
- Ensure that new code is tested and documented.
- Pull Requests will only be merged into main after review and approval by the repository owner.

---

## License

Released under the MIT License.

---

## Acknowledgements

- Based on the ATES/GWHC evaluation framework developed by Prof. Matthew Jackson's research group at Imperial College London.
- Inspired by Oracle Crystal Ball methodology, implemented in an open-source form.
- This tool was developed as part of the MSc Environmental Data Science and Machine Learning programme at Imperial College London.
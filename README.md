# Automated 3D Mold Generation

A tool for generating 3D molds from input STL files using advanced geometry operations. This project leverages Python libraries such as [trimesh](https://trimsh.org/), [Open3D](http://www.open3d.org/), and [ipywidgets](https://ipywidgets.readthedocs.io/) to process, repair, and transform 3D models. The tool is designed for research and industrial applications, facilitating rapid mold design for casting, injection molding, and prototyping.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Interactive Notebook](#interactive-notebook)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Overview

In many manufacturing processes, mold design is a critical and time-consuming step. This tool automates mold generation by:
- Computing a padded bounding box for a given watertight mesh.
- Subtracting the input mesh from the mold box to preserve the cavity.
- Optionally repairing imperfect meshes using enhanced Open3D routines.
- Applying a configurable draft angle to ease demolding.
- Adding user-specified vent/injection holes.
- Splitting the final mold into halves or quarters.

This makes it ideal for projects and rapid prototyping in academic or industrial settings.

## Features

- **Enhanced Mesh Repair:**  
  Automatically repairs non-watertight meshes using Open3D functionalities (e.g., removal of degenerate triangles, duplicate vertices, and non-manifold edges).

- **Draft Angle Integration:**  
  Applies a user-defined draft angle to taper the mold cavity, improving demolding ease.

- **Hole Processing:**  
  Supports user-specified vent or injection holes (e.g., bottom, top, left, right).

- **Flexible Splitting Modes:**  
  Splits the generated mold into halves or quarters for easier part removal.

- **Interactive Interface:**  
  Includes Jupyter Notebook demos with ipywidgets for interactive parameter tuning.

- **Command-Line Interface:**  
  Run the tool from the terminal with various command-line options for batch processing.

## Installation

### Requirements

- Python 3.7+
- [trimesh](https://trimsh.org/)  
- [numpy](https://numpy.org/)  
- [Open3D](http://www.open3d.org/)  
- [tqdm](https://tqdm.github.io/)  
- [ipywidgets](https://ipywidgets.readthedocs.io/) (optional, for interactive notebooks)

### Setup

1. **Clone the Repository:**

   ```bash
   git clone 
   cd mold-generator
   ```

2. **Install Dependencies:**

   It's recommended to use a virtual environment. Then run:

   ```bash
   pip install -r requirements.txt
   ```

   Your `requirements.txt` should include entries like:

   ```
   trimesh
   numpy
   open3d
   tqdm
   ipywidgets
   ```

## Usage

### Command-Line Interface

The tool can be run directly from the terminal. For example, to process a single STL file:

```bash
python src/mold_generator.py --input models/MAOI03b.stl --output output2101 --padding 0.1 --hole_positions bottom --split_mode quarters --draft_angle 0.0 --visualize
```

Or, to process all STL files in a directory:

```bash
python src/mold_generator.py --input models/ --output output2101 --padding 0.1 --hole_positions bottom --split_mode quarters --draft_angle 0.0
```

Use `python src/mold_generator.py -h` for a full list of options.

### Interactive Notebook

A demonstration notebook is available in the `notebooks/` folder. Open `notebooks/mold_generator.ipynb` in Jupyter Notebook to interactively modify parameters using ipywidgets and visualize the results.

## Repository Structure

```
mold-generator/
├── README.md                # Project overview and instructions
├── LICENSE                  # License file
├── requirements.txt         # Python package dependencies
├── .gitignore               # Files to ignore in Git
├── src/                     # Production code
│   └── mold_generator.py    # Main Python module with tool functions
├── notebooks/               # Jupyter notebooks for demos and experiments
│   └── mold_generator.ipynb # Interactive demonstration notebook
└── docs/                    # Additional documentation
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research, please consider citing it as follows:

```
Vasileios Papageridis, "Automated 3D Mold Generation Tool", GitHub repository, https://github.com/YourUsername/mold-generator.
```

## Acknowledgments

This project was developed as part of a research initiative supported by Institute of Computer Science - Foundation for Research and Technology Hellas (FORTH). Special thanks to the developers of trimesh, Open3D, and ipywidgets for providing the essential libraries used in this project.
```
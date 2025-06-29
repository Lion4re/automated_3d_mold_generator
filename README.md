# STL Mold Maker

Create professional negative space molds from STL files with intelligent wall thickness detection, alignment keys, and pour spouts. Perfect for casting resin, wax, chocolate, soap, and other materials.

## 🚀 Features

### Core Functionality
- **Intelligent Wall Thickness**: Automatically calculates optimal wall thickness based on object size, geometry, and manufacturing constraints
- **Multiple Mold Types**: 2-piece or 4-piece molds with customizable split axes
- **Smart Alignment Keys**: Corner-first positioning algorithm ensures perfect mold alignment
- **Pour Spouts**: Automatically positioned vertical spouts for easy material pouring
- **Draft Angles**: Optional draft angles (0.5-3.0°) for easier demolding

### Advanced Features
- **Enhanced Mesh Repair**: Multiple repair strategies (trimesh, Open3D, hybrid) for problematic STL files
- **Size-Adaptive Processing**: Different strategies for extra-small, small, medium, and large objects
- **Interactive Interface**: User-friendly guided setup with default values
- **Command Line Interface**: Full automation support for batch processing
- **Comprehensive Logging**: Detailed technical logs for troubleshooting and analysis
- **Settings Documentation**: Auto-generated settings files with printing tips

### File Support
- **Input**: STL files (binary and ASCII)
- **Output**: STL mold pieces, settings summary, detailed logs
- **Directory Structure**: Organized output with descriptive naming

## 📋 Requirements

### Required Dependencies
```bash
pip install trimesh numpy scipy opencv-python
```

### Optional Dependencies (Recommended)
```bash
# For advanced mesh repair
pip install open3d

# For enhanced visualization (optional)
pip install pyglet
```

### System Requirements
- Python 3.7 or higher
- Windows, macOS, or Linux
- 4GB+ RAM (for large STL files)

## 🛠️ Installation

1. **Download the STL Mold Maker**
   ```bash
   git clone https://github.com/your-repo/STL_Mold_Maker.git
   cd STL_Mold_Maker
   ```

2. **Install Dependencies**
   ```bash
   pip install trimesh numpy scipy opencv-python open3d
   ```

3. **Verify Installation**
   ```bash
   cd Python
   python makeMold.py
   ```

## 🎯 Quick Start

### Interactive Mode (Recommended for Beginners)

Simply run the program and follow the guided setup:

```bash
python makeMold.py
```

**Example Interactive Session:**
```
🏭 STL MOLD MAKER - Interactive Setup
💡 Tip: Press ENTER to use default values shown in parentheses

📁 Available STL Models (3 found):
 1. chess_knight.stl          (1.2 MB)
 2. miniature_house.stl       (856.3 KB)  
 3. decorative_vase.stl       (2.1 MB)

📋 Step 1: Select your STL model
Enter model number (1-3): 1
✅ Selected: chess_knight.stl

📋 Step 2: Configure mold parameters
Wall thickness (mm) - leave empty for auto-calculation (default: auto): 
Split axis (x=left-right, y=front-back, z=top-bottom) [x/y/z] (default: x): z
Number of mold pieces [2/4] (default: 2): 2
Number of alignment keys [2/4] (default: 2): 2
Mesh repair method [auto/trimesh/open3d/hybrid/none] (default: auto): 
Draft angle in degrees (0.5-3.0 for easier demolding, empty for none) (default: none): 1.5

📋 Configuration Summary:
📦 Model: chess_knight.stl
🔧 Wall thickness: Auto-calculated
⚡ Split axis: Z-axis
🧩 Mold pieces: 2
🔑 Alignment keys: 2
🛠️ Mesh repair: auto
📐 Draft angle: 1.5°

✅ Proceed with mold creation? [Y/n]: y
```

### Command Line Mode (Advanced Users)

For automation and batch processing:

```bash
# Basic 2-piece mold
python makeMold.py chess_knight.stl

# Custom configuration
python makeMold.py chess_knight.stl --wall_thickness 3.0 --split_axis z --mold_pieces 2 --num_alignment_keys 4

# 4-piece mold with draft angles
python makeMold.py large_sculpture.stl --mold_pieces 4 --draft_angle 2.0 --repair_method open3d
```

## 📚 Usage Examples

### Example 1: Simple Chess Piece Mold

**Input:** `chess_knight.stl` (small object, 25mm tall)

```bash
python makeMold.py chess_knight.stl --split_axis z
```

**Output:**
- `output/chess_knight_2-part/chess_knight_mold_bottom.stl`
- `output/chess_knight_2-part/chess_knight_mold_top.stl`
- `output/chess_knight_2-part/chess_knight_2-part_settings.txt`

**Result:** 2-piece mold split horizontally, 2.1mm wall thickness (auto-calculated), 2 alignment keys, vertical pour spout.

### Example 2: Large Decorative Object

**Input:** `decorative_vase.stl` (large object, 150mm tall)

```bash
python makeMold.py decorative_vase.stl --mold_pieces 4 --split_axis z --num_alignment_keys 4 --draft_angle 1.0
```

**Output:**
- `output/decorative_vase_4-part/decorative_vase_mold_bottom_left.stl`
- `output/decorative_vase_4-part/decorative_vase_mold_bottom_right.stl`
- `output/decorative_vase_4-part/decorative_vase_mold_top_left.stl`
- `output/decorative_vase_4-part/decorative_vase_mold_top_right.stl`

**Result:** 4-piece mold for easier handling of large object, 1° draft angles for easier demolding.

### Example 3: Problematic STL File

**Input:** `broken_model.stl` (non-watertight mesh with holes)

```bash
python makeMold.py broken_model.stl --repair_method hybrid
```

**Process:**
1. Detects non-watertight mesh
2. Applies hybrid repair (trimesh + Open3D)
3. Creates mold with repaired geometry
4. Logs all repair steps for review

### Example 4: Tiny Miniature

**Input:** `miniature_detail.stl` (very small object, 8mm)

```bash
python makeMold.py miniature_detail.stl --split_axis y
```

**Automatic Adaptations:**
- Extra-small object detection
- Reduced wall thickness (1.8mm)
- Smaller alignment keys (0.3mm radius)
- Conservative safety margins
- Precise positioning algorithms

## ⚙️ Configuration Options

### Split Axes
- **X-axis**: Left-right split (good for tall objects)
- **Y-axis**: Front-back split (good for long objects)
- **Z-axis**: Top-bottom split (good for flat objects)

### Wall Thickness
- **Auto**: Intelligent calculation based on object size (recommended)
- **Manual**: Specify exact thickness in millimeters
- **Range**: 1.5mm - 25mm (automatically constrained)

### Mold Pieces
- **2-piece**: Simpler printing and assembly
- **4-piece**: Better for large objects, easier handling

### Alignment Keys
- **2 keys**: Diagonal placement for optimal stability
- **4 keys**: Maximum alignment precision

### Draft Angles
- **None**: Most accurate reproduction
- **0.5-1.0°**: Slight taper for easier demolding
- **1.5-3.0°**: Easier demolding, less precision

### Mesh Repair Methods
- **auto**: Smart detection and repair (recommended)
- **trimesh**: Basic built-in repair
- **open3d**: Advanced repair algorithms
- **hybrid**: Both trimesh and Open3D
- **none**: Skip repair (use only for perfect meshes)

## 📁 Output Files

Each mold generation creates an organized output directory:

```
output/
└── model_name_2-part/
    ├── model_name_mold_bottom.stl    # Bottom mold piece
    ├── model_name_mold_top.stl       # Top mold piece  
    └── model_name_2-part_settings.txt # Complete settings summary

logs/
└── moldmaker_model_name_20240101_123456.log  # Detailed technical log
```

### Settings File Example
```
STL MOLD MAKER - SETTINGS SUMMARY
Generated: 2024-01-01 12:34:56

INPUT OBJECT
File: chess_knight.stl
Dimensions: 23.1 x 15.6 x 25.3 mm
Volume: 2847 mm³
Size Category: SMALL

MOLD CONFIGURATION
Type: 2-piece mold
Split Axis: Z-axis
Wall Thickness: 2.10 mm (CALCULATED)
User Input: Auto-calculated
Alignment Keys: 2
Mesh Repair: auto
Draft Angles: 1.5 degrees (Z-axis)

OUTPUT FILES
- chess_knight_mold_bottom.stl
- chess_knight_mold_top.stl

PRINTING TIPS
- Use 0.1-0.2mm layer height for smooth surfaces
- Print with 100% infill for strength
- Spray with oil or silicone spray to reduce friction
- Pour material through bottom spout
- Alignment keys ensure proper assembly
```

## 🖨️ 3D Printing Tips

### Print Settings
- **Layer Height**: 0.1-0.2mm for smooth cavity surfaces
- **Infill**: 100% for maximum strength
- **Support**: Usually not needed (molds print cavity-up)
- **Print Speed**: Moderate (40-60mm/s) for quality

### Materials
- **PLA**: Easy printing, good for low-temp casting
- **PETG**: Chemical resistance, higher temperature tolerance
- **ABS**: Acetone smoothing, very smooth surfaces
- **ASA**: UV resistance for outdoor use

### Post-Processing
1. **Remove supports** if any were used
2. **Light sanding** of parting lines
3. **Apply release agent** (silicone spray, oil, or soap)
4. **Test fit** mold pieces before casting

## 🔧 Troubleshooting

### Common Issues

**"No STL files found"**
- Place STL files in current directory, `models/`, or `Python/models/`
- Check file extensions (.stl)

**"Mesh is not watertight"**
- Use `--repair_method auto` or `open3d`
- Check log files for repair details
- Consider manual mesh repair in Blender/Fusion360

**"Alignment keys too close to cavity"**
- Object might be too complex for automatic positioning
- Try different split axis
- Check log files for detailed positioning analysis

**"Boolean operation failed"**
- Usually caused by mesh issues
- Try different repair methods
- Simplify original STL if possible

**Out of memory errors**
- Reduce STL file complexity
- Close other applications
- Use 64-bit Python

### Getting Help

1. **Check log files** in `logs/` directory for detailed error information
2. **Try different settings** (repair method, split axis)
3. **Simplify STL file** if very complex
4. **Report issues** with minimal example and log files

## 🎨 Advanced Examples

### Batch Processing Script

```python
import os
import subprocess

stl_files = ['piece1.stl', 'piece2.stl', 'piece3.stl']

for stl_file in stl_files:
    cmd = [
        'python', 'makeMold.py', stl_file,
        '--mold_pieces', '2',
        '--draft_angle', '1.0',
        '--repair_method', 'auto'
    ]
    subprocess.run(cmd)
    print(f"Completed {stl_file}")
```

### Custom Wall Thickness by Size

```bash
# Small objects (< 20mm): thin walls
python makeMold.py small_detail.stl --wall_thickness 1.5

# Medium objects (20-50mm): standard walls  
python makeMold.py medium_part.stl --wall_thickness 3.0

# Large objects (> 50mm): thick walls
python makeMold.py large_sculpture.stl --wall_thickness 5.0
```

### Complex Multi-Part Projects

```bash
# Create complementary molds for a multi-part assembly
python makeMold.py part_A.stl --split_axis x --mold_pieces 2
python makeMold.py part_B.stl --split_axis x --mold_pieces 2  
python makeMold.py part_C.stl --split_axis y --mold_pieces 4
```

## 🏆 Best Practices

1. **Start with interactive mode** to understand options
2. **Use auto wall thickness** unless you have specific requirements
3. **Choose split axis** based on object geometry and desired parting line
4. **Enable draft angles** for easier demolding (1-2° is usually sufficient)
5. **Use mesh repair** unless you're certain your STL is perfect
6. **Print test pieces** with cheap filament first
7. **Keep detailed records** using the generated settings files

## 📊 Specifications

- **Maximum STL Size**: Limited by available RAM (tested up to 50MB files)
- **Minimum Object Size**: 5mm (smaller objects may need manual thickness)
- **Wall Thickness Range**: 1.5mm - 25mm
- **Draft Angle Range**: 0.5° - 3.0°
- **Alignment Key Count**: 2 or 4
- **Split Axes**: X, Y, or Z
- **Mold Pieces**: 2 or 4

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with various STL files
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Trimesh](https://trimesh.org/) for mesh processing
- [Open3D](https://open3d.org/) for advanced mesh repair
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computations
- [OpenCV](https://opencv.org/) for image processing in EDT algorithms
# Mulimodal Floorplan Encoding
Aligning rasterized floorplans under multiple modalities (architect drawing, projected SfM point cloud to 2D, noisy floorplan predicted by another method, etc) in a shared latent space. This latent space can be leveraged for downstream applications.

## Installation

### Basic Installation
```bash
conda create -n mmfe python=3.11
conda activate mmfe
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

### Optional RoMa
```
curl -fsSL https://astral.sh/uv/install.sh | sh
uv pip install -e .[roma]
uv pip install -e .[roma1]
```

**Troubleshooting**
```
RoMa: ModuleNotFoundError: No module named 'local_corr'
```

If you install RoMa and receive a ModuleNotFoundError for local_corr, it is likely due to a PyTorch version incompatibility. ([RoMa Issue](https://github.com/Parskatt/RoMa/issues/131))

Fix (for PyTorch 2.9.0+): You must install a compatible wheel file.

Uninstall the broken package:

```
pip uninstall fused-local-corr
```

Install the correct wheel:

* Go to [this GitHub Actions run.](https://github.com/Parskatt/fused-local-corr/actions/runs/18772643426) (You must be logged into GitHub to download).

* Download the "wheels" artifact at the bottom of the page.

* Unzip the file and install the wheel matching your Python version (e.g., cp311 for Python 3.11). Make sure your conda environment is activated.

    ```
    pip install /path/to/unzipped/wheels/fused_local_corr-0.0.1-cp311-cp311-linux_x86_64.whl
    ```


### CubiCasa Installation

The project includes CubiCasa5K, a large-scale floorplan image dataset and model. There are two ways to install CubiCasa:

```bash
# Navigate to the CubiCasa directory
cd src/third_parties/CubiCasa5k

# Install CubiCasa requirements
pip install -r requirements.txt

# Install OpenCV (if not already installed)
pip install opencv-python==3.1.0
```

## Datasets

### Download CubiCasa5K Dataset

### Download Structured3D Dataset


### Download Swiss Dwellings Dataset


### Download Zillow Indoor Dataset

Follow the steps indicated in the original (Zillow Indoor)[https://github.com/zillow/zind] repository. 
Clone the repository and install the environment. We recommend to do this in a different environment from *mmfe*.
Replace the *render.py* and *visualize_zind_cli.py* by the versions in *mmfe/src/data_preprocess/zillow*.

Once you have downloaded the data as indicated in the repository and changed the above mentioned files run the following command:
```bash
python code/visualize_zind_cli.py --input ./data --output ./outputs --render-mmfe --redraw
```

This should give you a hierarchy like this:

<pre> outputs/
└── render_data/
    ├── 0000/
    │   ├── overlay/
    │   │   └── raster_to_vector_redraw_layout_floor_01.jpg
    │   └── vector/
    │       └── vector_only_redraw_layout_floor_01.jpg
    ├── 0001/
    │   ├── overlay/
    │   │   ├── raster_to_vector_redraw_layout_floor_01.jpg
    │   │   └── raster_to_vector_redraw_layout_floor_02.jpg
    │   └── vector/
    │       ├── vector_only_redraw_layout_floor_01.jpg
    │       └── vector_only_redraw_layout_floor_02.jpg
    ├── ...
    └── 1574/
        ├── overlay/
        │   ├── raster_to_vector_redraw_layout_floor_01.jpg
        │   └── raster_to_vector_redraw_layout_floor_02.jpg
        └── vector/
            ├── vector_only_redraw_layout_floor_01.jpg
            └── vector_only_redraw_layout_floor_02.jpg
 </pre>

Then run the following script:
```bash
python scripts/datasets_scripts/bring_zind.py --source-dir PATH_TO_ZIND_REPO/zind/outputs/render_data/ --target-dir PATH_TO_MMFE_DATA/data/zind --partition-file PATH_TO_ZIND_REPO/zind/zind_partition.json
```


### Download ScanNet Dataset (Crossover Format)

Follow the steps indicated in the original (Crossover)[https://github.com/GradientSpaces/CrossOver] repository. 

1. Download ScanNetV2 version


## Usage

### Basic Floorplan Generation
```python
from mmfe_utils.cubicasa import scene_to_floorplan_image

# Generate a floorplan image
img = scene_to_floorplan_image("path/to/data", scene_id=1, no_color=False, bbox_percentage=0.5)
```

### LiDAR Simulation
```python
from mmfe_utils.cubicasa import scene_to_lidar_image

# Generate a simulated LiDAR scan
lidar_img = scene_to_lidar_image("path/to/data", scene_id=1, no_color=False, bbox_percentage=0.5)
```

### Hand-Drawn Simulation
```python
from mmfe_utils.cubicasa import scene_to_handdrawn_image

# Generate a hand-drawn floorplan simulation
handdrawn_img = scene_to_handdrawn_image("path/to/data", scene_id=1, no_color=False, bbox_percentage=0.5)
```


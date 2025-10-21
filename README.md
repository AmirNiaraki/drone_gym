# Drone Gym: A Simulation Environment for Power and Perception-aware 3D Path Planning with Reinforcement Learning

A Gymnasium-compatible simulation environment built on stable-baselines3 for training and evaluating reinforcement learning algorithms for autonomous drone path planning. This environment supports the IP4 (Integrated Power and Perception Path Planning) framework for aerial detection of organic objects in non-exhaustively searchable survey areas.

**Paper**: [Maximizing aerial detection of organic objects in non-exhaustively searchable survey area](https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/Niaraki_Maximizing_aerial_detection_of_organic_objects_in_non-exhaustively_searchable_survey_CVPRW_2025_paper.html) - CVPR 2025 Workshop

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [Citation](#citation)

## 🎯 Overview

This simulation environment provides a comprehensive framework for developing and testing drone path planning algorithms. The environment includes:

- **Gymnasium-compatible RL environment** with discrete and continuous action spaces
- **Physics-based flight dynamics** with wind field integration and battery constraints
- **Multiple navigation strategies** including keyboard control and complete coverage path planning
- **Object detection integration** supporting RetinaNet, clustering-based, and low-fidelity detection methods
- **Real-time visualization** with OpenCV for monitoring drone behavior

The environment was designed for agricultural anomaly detection using Near-Infrared (NIR) imaging, but can be adapted for various aerial surveillance and exploration tasks.

## ✨ Key Features

- **🚁 Multiple Navigation Modes**: Keyboard control (`keyboard_player.py`), Complete Coverage (`CCdrone.py`), and Hierarchical sampling
- **🧠 Detection Model Integration**: Low-fidelity simulation, RetinaNet, and Double-clustering methods
- **🌪️ Wind-Aware Physics**: Realistic drag modeling based on wind conditions and flight dynamics
- **🔋 Battery Constraints**: Power-aware navigation with configurable battery limits
- **📊 Real-time Visualization**: OpenCV-based monitoring with detection overlays
- **🎮 Interactive Testing**: Keyboard-based manual navigation for algorithm validation
- **📈 Comprehensive Logging**: Flight telemetry, detection data, and performance metrics

## 🏗️ Architecture

The system consists of four main components:

### 1. Environment (`drone_environment.py`)

- **Gymnasium-compatible environment** for RL training
- **3D navigation physics** with wind field integration
- **Real-time rendering** with OpenCV visualization
- **Battery and power modeling** for realistic flight constraints

### 2. Navigation (`navigator.py`, `CCdrone.py`, `keyboard_player.py`)

- **CompleteCoverageNavigator**: Traditional grid-based coverage pattern
- **KeyboardNavigator**: Manual control for testing and demonstration
- **HierarchicalNavigator**: Multi-altitude sampling strategy
- **CCdrone.py**: Standalone complete coverage implementation
- **keyboard_player.py**: Standalone keyboard control interface

### 3. Perception (`detector.py`, `inference.py`)

- **RetinaNet**: Deep learning object detection
- **Double-clustering**: KMeans + DBSCAN for anomaly detection
- **Low-fidelity**: Simple pixel counting for simulation

### 4. Training (`dronelearn.py`, `test_model.py`)

- **Stable-baselines3 integration** supporting A2C, PPO, and DQN algorithms
- **Multi-threaded learning** with parallel environment execution
- **Tensorboard logging** for training visualization and monitoring

## ⚠️ Important Notice

**Proprietary Components**: The following components are proprietary and require permission for use:

1. **Datasets**: The NIR datasets used for training are proprietary
2. **Object Detection Models**: Pre-trained RetinaNet models are proprietary
3. **RL Models**: Trained reinforcement learning models are proprietary

**Available for Research**: The double-clustering method for object detection on NDVI images is available for research use.

**Collaboration**: For access to proprietary components or collaboration opportunities, please contact the authors.

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for RetinaNet)
- 8GB+ RAM

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd drone_gym
```

2. **Create virtual environment**

```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
# or
myenv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install PyTorch (if using RetinaNet)**

```bash
pip install torch torchvision torchaudio
```

## 🎮 Quick Start

### Basic Navigation Demo

```bash
# Keyboard-controlled navigation
python main.py --image_path images/sample.png --navigator keyboard --detector low_fidelity

# Complete coverage path planning
python main.py --image_path images/sample.png --navigator complete --detector retina

# Standalone keyboard control
python keyboard_player.py

# Standalone complete coverage
python CCdrone.py
```

### Interactive World Creation

```bash
# Create custom world with mouse drawing
python drawer.py --name my_world
```

### Training RL Agents

```bash
# Train A2C agent with stable-baselines3
python dronelearn.py

# Test trained model
python test_model.py
```

## 📖 Usage Examples

### 1. Manual Navigation with Object Detection

```python
from drone_environment import droneEnv
from navigator import KeyboardNavigator
from inference import Inferer

# Initialize environment
env = droneEnv(observation_mode="cont", action_mode="cont", render=True,
               img_path="images/field.png")

# Setup navigation and detection
navigator = KeyboardNavigator(env)
detector = Inferer(env.cfg, model_type="retina")

# Navigate and detect
for obs, info in navigator.navigate():
    score, boxes = detector.infer(obs)
    print(f"Detected {len(boxes)} objects with score {score}")
```

### 2. Complete Coverage Path Planning

```python
from navigator import CompleteCoverageNavigator

# Initialize navigator
navigator = CompleteCoverageNavigator(env)

# Execute coverage pattern
for obs, info in navigator.navigate():
    # Process detections
    score, boxes = detector.infer(obs)
    print(f"Step {info['step_count']}: Battery {info['battery_levels']}%")
```

### 3. Hierarchical Multi-Altitude Sampling

```python
from navigator import HierarchicalNavigator

# Setup hierarchical navigation
navigator = HierarchicalNavigator(env)
navigator.edge_discretization_segments = 5  # 5x5 grid
navigator.sampling_velocity = 10  # m/s

# Execute sampling
for obs, info in navigator.navigate():
    # Analyze detections at different altitudes
    score, boxes = detector.infer(obs)
```

### 4. Training Custom RL Agent

```python
from stable_baselines3 import A2C
from drone_environment import droneEnv

# Create environment
env = droneEnv(observation_mode='disc', action_mode='cont', render=True)

# Train A2C agent
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)

# Save trained model
model.save("my_drone_agent")
```

## 📁 Project Structure

```
drone_gym/
├── main.py                     # Main entry point
├── drone_environment.py       # Core RL environment
├── navigator.py               # Navigation strategies
├── detector.py                # Object detection models
├── inference.py               # Detection inference engine
├── configurations.py          # Environment configuration
├── model_loader.py            # Model loading utilities
├── image_resizer.py           # Image preprocessing utilities
├── plots.py                   # Flight data visualization
├── drawer.py                  # Interactive world creation
├── dronelearn.py             # RL training script
├── test_model.py             # Model testing
├── keyboard_player.py        # Manual control
├── pytorch-retinanet/        # RetinaNet implementation
├── weights/                   # Pre-trained models
├── images/                    # Sample images and worlds
├── input_images/             # Training datasets (field1-field12)
├── flight_logs/              # Flight data logs
└── requirements.txt          # Dependencies
```

## ⚙️ Configuration

### Environment Settings (`configurations.py`)

```python
# Flight parameters
FOV_X = 60/2  # Horizontal field of view (degrees)
FOV_Y = 60/2  # Vertical field of view (degrees)
FRAME_W = 320  # Camera resolution width
FRAME_H = 320  # Camera resolution height

# Flight constraints
min_flight_height = 60   # meters
max_flight_height = 180  # meters
MAX_SPEED = 5           # m/s
FULL_BATTERY = 100      # battery units

# Wind and physics
DEFAULT_WIND = (3.5, 0.)  # m/s (east, north)
drag_table = "drag_dataset.csv"  # CFD-based drag coefficients
```

### Detection Models

**RetinaNet Configuration:**

```python
model_config = ModelConfig(
    depth=50,           # ResNet backbone depth
    num_classes=1,      # Number of object classes
    model_path="weights/best_anomaly.pt"
)
```

**Double-clustering Parameters:**

```python
# KMeans clustering
n_clusters = 4
selected_label = 2  # Anomaly cluster

# DBSCAN clustering
eps = 10           # Maximum distance between samples
min_samples = 300  # Minimum samples per cluster
```

## 🎓 Training

### 1. Prepare Training Data

```bash
# Create custom world
python drawer.py --name training_world

# Or use existing orthomosaic
cp your_field.png images/field.png
```

### 2. Train RL Agent

```python
# Basic training
python dronelearn.py

# Custom training parameters
python -c "
from stable_baselines3 import A2C
from drone_environment import droneEnv

env = droneEnv(observation_mode='disc', action_mode='cont', render=True)
model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.0005)
model.learn(total_timesteps=2000000)
model.save('trained_agent')
"
```

### 3. Train Object Detection Models

```bash
# Train double-clustering detector (available for research)
python detector.py

# Process multiple field images (field1-field12)
# Automatically generates segmented images for each field

# Note: RetinaNet training requires proprietary datasets
# Contact authors for access to pre-trained models
```

## 🔍 Inference

### Real-time Detection

```python
# Load trained model
from stable_baselines3 import A2C
model = A2C.load("trained_agent")

# Run inference
obs = env.reset()
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, info = env.step(action)

    # Detect objects
    score, boxes = detector.infer(obs)
    print(f"Detected {len(boxes)} anomalies")
```

### Post-processing Analysis

```bash
# Analyze flight data
python main.py --is_post_process

# Visualize flight performance
python plots.py

# This generates:
# - output.png: Image with detection overlays
# - data_info.json: Flight telemetry
# - flight_logs/: Detailed flight data
# - Performance plots: Step counts and detection scores over time
```

## 📊 Results

Based on the research paper ([CVPR 2025 Workshop](https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/Niaraki_Maximizing_aerial_detection_of_organic_objects_in_non-exhaustively_searchable_survey_CVPRW_2025_paper.html)), the IP4 framework demonstrates:

### Performance Comparison

| Method         | Wind Speed  | Surveyed Area (m²) | Objects Detected | Flight Time |
| -------------- | ----------- | ------------------ | ---------------- | ----------- |
| CCPP 60m       | 2.2 m/s     | 9,310              | 17               | 21 min      |
| CCPP 120m      | 2.7 m/s     | 32,300             | 24               | 20 min      |
| **IP4 (Ours)** | **2.6 m/s** | **16,600**         | **36**           | **24 min**  |

### Key Achievements

- **3x more detections** than traditional CCPP methods
- **Wind-aware navigation** with physics-based drag modeling
- **Real-time processing** with 0.04s inference time (vs 0.656s baseline)
- **Adaptive altitude** selection for optimal coverage

## 🛠️ Advanced Usage

### Custom Wind Conditions

```python
# Set wind field
env.wind = (5.0, -2.0)  # 5 m/s east, 2 m/s south

# Load real-time wind data
import requests
wind_data = requests.get("weather_api_url").json()
env.wind = (wind_data['east'], wind_data['north'])
```

### Multi-Agent Training

```python
# Parallel training with multiple agents
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return droneEnv(observation_mode='disc', action_mode='cont')

env = DummyVecEnv([make_env for _ in range(8)])
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000000)
```

### Custom Detection Models

```python
# Add new detector
class CustomDetector(BaseDetector):
    def infer(self, image):
        # Your detection logic
        boxes = your_detection_algorithm(image)
        return boxes, scores

# Register detector
detector = Inferer(env.cfg, model_type="custom")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **OpenCV Display Issues**

   ```bash
   # For headless systems
   export DISPLAY=:0
   # Or use virtual display
   xvfb-run -a python main.py
   ```

3. **Model Loading Errors**
   ```python
   # Check model path and weights
   import torch
   model = torch.load("weights/best_anomaly.pt", map_location='cpu')
   ```

## 🤝 Contributing

### Open Source Components

- **Simulation Environment**: The core Gymnasium environment is open source
- **Double-clustering Detection**: Available for research and development
- **Navigation Algorithms**: Keyboard control and complete coverage implementations
- **Data Visualization**: Flight performance plotting and analysis tools
- **Image Processing**: Resizing and preprocessing utilities

### Proprietary Components

- **Datasets**: NIR training datasets require permission
- **Pre-trained Models**: RetinaNet and RL models are proprietary
- **Collaboration**: Contact authors for access to proprietary components

### Contributing Guidelines

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

**Note**: Contributions should focus on the open-source simulation environment, double-clustering detection methods, and data visualization tools.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Niaraki_2025_CVPR,
  author = {Niaraki, Amir and Herrera-Gerena, Jansel and Roghair, Jeremy and Jannesari, Ali},
  title = {Maximizing aerial detection of organic objects in non-exhaustively searchable survey area},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2025},
  pages = {5491-5499}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Stable Baselines3** for reinforcement learning algorithms
- **PyTorch RetinaNet** for object detection implementation
- **OpenCV** for computer vision and visualization
- **Gymnasium** for RL environment framework
- **CVPR 2025 Workshop** for paper publication

---

**Note**: This simulation environment is designed for research and educational purposes. The double-clustering detection method is available for research use. For access to proprietary datasets, pre-trained models, or collaboration opportunities, please contact the authors. Ensure compliance with local aviation regulations when deploying in real-world scenarios.

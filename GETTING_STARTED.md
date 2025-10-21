# Getting Started with Drone Gym

This guide will help you set up and start using the Drone Gym simulation environment for UAV path planning with reinforcement learning.

## 🚀 Quick Installation

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

# Generate random pixelated world
# The environment automatically generates worlds with configurable parameters
```

### World Generation Examples

```python
# Generate world with custom parameters
from drone_environment import droneEnv

# Create environment with random world generation
env = droneEnv(observation_mode="cont", action_mode="cont", render=True)

# World parameters are configured in configurations.py:
# - SEEDS: Number of target objects (default: 200)
# - square_size_range: Size range of targets (default: 1-10 pixels)
# - world_size: World dimensions (default: 2000x500 pixels)
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

---

**Next Steps**: Check out the [main README](../README.md) for project overview and architecture details.

# Drone Gym: A Simulation Environment for Power and Perception-aware 3D Path Planning with Reinforcement Learning

A Gymnasium-compatible simulation environment built on stable-baselines3 for training and evaluating reinforcement learning algorithms for autonomous drone path planning. This environment supports the IP4 (Integrated Power and Perception Path Planning) framework for aerial detection of organic objects in non-exhaustively searchable survey areas.

## 🎯 Overview

This simulation environment provides a comprehensive framework for developing and testing drone path planning algorithms. The environment includes:

- **Gymnasium-compatible RL environment** with discrete and continuous action spaces
- **Physics-based flight dynamics** with wind field integration and battery constraints
- **Multiple navigation strategies** including keyboard control and complete coverage path planning
- **Object detection integration** supporting RetinaNet, clustering-based, and low-fidelity detection methods
- **Real-time visualization** with OpenCV for monitoring drone behavior

The environment was designed for agricultural anomaly detection using Near-Infrared (NIR) imaging, but can be adapted for various aerial surveillance and exploration tasks.

## 🌍 World Generation and Reward System

### Pixelated World Environment

The simulation includes a **world generator** that creates synthetic environments for training and testing:

- **Random Seed Generation**: Creates black pixel targets (anomalies) on white background
- **Configurable World Size**: Customizable dimensions and target density
- **Reward System**: Black pixel detection as reward signal for RL training
- **Physics Integration**: Wind-aware navigation with realistic drag modeling

### Reward Mechanism

The environment uses a **black pixel counting reward system**:

- **Detection Reward**: Number of black pixels in drone's field of view
- **Power Cost**: Battery consumption based on wind conditions and movement
- **Coverage Reward**: Positive reward for visiting new areas
- **Duplicate Penalty**: Negative reward for revisiting locations

This pixelated world approach allows for:

- **Rapid Training**: Fast simulation without real-world data requirements
- **Scalable Testing**: Easy generation of diverse scenarios
- **Research Flexibility**: Adaptable to different target detection tasks

## ✨ Key Features

- **🚁 Multiple Navigation Modes**: A2C RL policy (`dronelearn.py`),Complete Coverage (`CCdrone.py`) and Keyboard control (`keyboard_player.py`)
- **🧠 Detection Model Integration**: Low-fidelity simulation, RetinaNet, and Double-clustering methods
- **🌪️ Wind-Aware Physics**: Realistic drag modeling based on wind conditions and flight dynamics
- **🔋 Battery Constraints**: Power-aware navigation with configurable battery limits
- **📊 Real-time Visualization**: OpenCV-based monitoring with detection overlays
- **🎮 Interactive Testing**: Keyboard-based manual navigation for algorithm validation

## 🛠️ Advanced Usage

For advanced configuration, training, and troubleshooting, see the **[Getting Started Guide](GETTING_STARTED.md)**.

## 🤝 Contributing

This simulation environment is designed for research and educational purposes. The environment operates on a **pixelated world** where the reward is based on detecting black pixels on a white background of custom size. The double-clustering detection method can be implemented on other NIR datasets for research use. For access to proprietary datasets, pre-trained RL and object detection models, or collaboration opportunities, please contact the authors. Ensure compliance with local aviation regulations when deploying in real-world scenarios.

## 📚 Citation

**Paper**: [Maximizing aerial detection of organic objects in non-exhaustively searchable survey area](https://openaccess.thecvf.com/content/CVPR2025W/V4A/html/Niaraki_Maximizing_aerial_detection_of_organic_objects_in_non-exhaustively_searchable_survey_CVPRW_2025_paper.html) - CVPR 2025 Workshop

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

# Drone Gym

_A Simulation Environment for Power and Perception-aware 3D Path Planning with Reinforcement Learning_

<div align="center">
<table>
<tr>
<td align="center">
<img src="sample%20images/CCPP.gif" alt="Complete Coverage Path Planning" height="200">
<p><em>Complete Coverage Path Planning (CCPP)</em></p>
</td>
<td align="center">
<img src="sample%20images/IP4_exploring.gif" alt="IP4 Path Planning" height="200">
<p><em>IP4 Path Planning</em></p>
</td>
</tr>
</table>
</div>

A Gymnasium-compatible simulation environment built on stable-baselines3 for training and evaluating reinforcement learning algorithms for autonomous drone path planning. This environment supports our IP4 (Integrated Power and Perception Path Planning) framework for aerial detection of organic objects in non-exhaustively searchable survey areas.

<div align="center">
<img src="sample%20images/RL%20structure%20drone%20gym.png" alt="Drone Gym Architecture" height="300">
</div>

The environmen is consructed such that the navigation module and perception module work asyncronously. So the environment can be used to generate a random or pixel world for Perception module first can operate on this low-fidelity environment with desired reward distribution. Then the target map can be replaced with real world geotiffs and perception module can be replaced with otehr object detection models of choice. Here, a double-clustering method and a DL method are implemented

<div align="center">
<table>
<tr>
<td align="center">
<img src="sample%20images/NDVI.png" alt="NDVI filtered image of grass fields" height="200">
<p><em>NDVI filtered image of grass fields</em></p>
</td>
<td align="center">
<img src="sample%20images/world_20241109-131120_seed200.png" alt="Pixelated World - 200 Seeds" height="200">
<p><em>Pixelated World with 200 Seeds</em></p>
</td>
<td align="center">
<img src="sample%20images/Retina.png" alt="Anomaly detection Retina-net" height="200">
<p><em>Anomaly detection with retina net</em></p>
</td>
</tr>
</table>
</div>

Which then can be replaced with actual NIR imagery of fields:

## ✨ Key Features

This simulation environment provides a comprehensive framework for developing and testing drone path planning algorithms. The environment includes:

- **Gymnasium-compatible RL environment** with discrete and continuous action spaces
- **Physics-based flight dynamics** with wind field integration and battery constraints
- **Multiple navigation strategies** including keyboard control and complete coverage path planning, Hierarchical Method and Actor-Critic Reinforcement learning
- **Object detection integration** supporting RetinaNet, clustering-based, and low-fidelity detection methods

The environment was designed for agricultural anomaly detection using Near-Infrared (NIR) imaging, but can be adapted for various aerial surveillance and exploration tasks.

For advanced configuration, training, and troubleshooting, see the **[Getting Started Guide](GETTING_STARTED.md)**.

## 🤝 Contributing

This simulation environment is designed for research and educational purposes. The environment is ready to use on a **pixelated world** where the reward is based on detecting black pixels on a white background of custom size. The double-clustering detection method can be implemented on other NIR datasets for research use. For access to proprietary datasets, pre-trained RL and object detection models, or collaboration opportunities, please contact the authors. Ensure compliance with local aviation regulations when deploying in real-world scenarios.

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

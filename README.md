# Smart Shopping Cart Robot

This project is a ROS 2-based intelligent shopping cart robot that combines indoor navigation and object detection to assist in smart logistics scenarios, such as autonomous shopping or item transport.

## Features

- 🔍 **YOLO Object Detection**: Real-time object recognition using a pretrained YOLO model (`best.pt`).
- 🗺️ **SLAM-Based Navigation**: Uses SLAM for mapping and autonomous navigation.
- 🤖 **ROS 2 Integration**: Developed as a ROS 2 package for modular and scalable robotics development.
- 🧭 **Predefined Map**: Includes `.pgm` and `.yaml` files for indoor localization.

## Project Structure

```
.vscode/                  # VSCode settings
map.pgm / map.yaml        # Map files for navigation
src/yollllllo/            # Main ROS 2 package
├── best.pt               # YOLO model weights
├── setup.py              # Python package setup
├── package.xml           # ROS 2 package manifest
```

## Installation

1. Clone this repository into your ROS 2 workspace:
   ```bash
   git clone https://github.com/yourusername/smart-shopping-cart-robot.git
   ```
2. Install dependencies and build:
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   colcon build
   ```

## Usage

1. Launch the robot navigation and object detection nodes:
   ```bash
   source install/setup.bash
   ros2 launch yollllllo bringup.launch.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

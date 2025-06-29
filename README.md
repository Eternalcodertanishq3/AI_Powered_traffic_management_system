# AI-Powered Traffic Management System

## Overview
This project implements an AI-powered traffic management system that simulates real-time traffic control using pre-recorded images from the KITTI Vision dataset. The system:
- Detects vehicles on each “frame” (KITTI image) using a pre-trained Faster R-CNN.
- Simulates a 4-lane intersection by splitting the detected vehicle count.
- Uses a multi-agent reinforcement learning (MARL) approach where one independent DQN agent (Deep Q-Network) per lane decides whether to toggle the traffic signal.
- Provides on-screen visualization with detection overlays, the current traffic state, and the actions selected by each agent.

This prototype lays the groundwork for future enhancements such as advanced MARL algorithms (e.g., MADDPG), integration with live video feeds, and eventual hardware deployment.

## Features
- **Automatic Dataset Handling:**  
  Uses Torchvision’s built-in `Kitti` class to automatically download (if needed) and load the KITTI dataset.
- **Vehicle Detection:**  
  Leverages a pre-trained Faster R-CNN model for detecting vehicles.
- **Traffic State Simulation:**  
  Distributes the overall vehicle count equally among 4 lanes and maintains signal states.
- **Multi-Agent RL:**  
  Implements independent DQN agents for each lane. Each agent makes decisions based on its local 2-dimensional state.
- **Real-Time Visualization:**  
  Overlays vehicle counts, state vectors, and agent-selected actions on the images.
- **Modular and Extensible:**  
  The system is organized into modular files making future enhancements straightforward.

## Project Structure

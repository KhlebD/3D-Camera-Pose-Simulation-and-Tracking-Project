# 3D Camera Pose Simulation and Tracking  
This project is an interactive 3D Camera Pose Simulation and Tracking System, developed as part of a 3D graphics course. The application visualizes camera movement, computes 3D poses, and allows users to interactively explore a simulated environment using OpenGL and OpenCV.  
![WhatsApp Image 2024-12-17 at 14 29 07](https://github.com/user-attachments/assets/3b47080e-c8b5-44a7-b646-418b11e4c808)

Features  
Interactive Terrain Simulation:  

Generates a dynamic 3D terrain with adjustable scale and detail.  
Visualizes camera positions and orientations in a 3D environment.  
Camera Pose Estimation:  

Computes camera pose using 2D-3D correspondences with OpenCV's solvePnP.  
Tracks both true and computed camera trajectories for comparison.  

User Interaction:  
 
Record and visualize camera movement through the environment.  
Add and manage 3D points interactively using mouse inputs.  
Toggle between different views and modes (e.g., tracker setup, normal, flight modes).  
3D Graphics Integration:  

Renders real-time 3D graphics using OpenGL.  
Supports picking, camera fly-through, and blended visualizations.  

Dependencies:  
OpenGL (via GLFW and GLAD)  
OpenCV (for pose estimation and image processing)  
GLM (for matrix operations)  
stb_image (for texture loading)  
Ensure these libraries are installed and linked to your project.  

Camera Movement:  

Arrow keys: Move camera (2D plane)  
< and > keys: Move up and down  
Key Features:  

Ctrl + M: Switch between Tracker Setup and Normal mode.  
Ctrl + B: Record current camera pose.  
Ctrl + E: Toggle estimated camera view.  
Mouse Interaction:  

Right-click: Add or delete points in Tracker Setup mode.  
Project Structure  
Main Code: Main.cpp - Contains the core logic for OpenGL rendering and camera tracking.  
Shaders: Vertex and fragment shaders for terrain and point visualization.  
Textures: Used for terrain and other graphical elements.  

Acknowledgments  
This project was developed as part of the 3D Graphics Mini-Project in Ben Gurion University (Prof. Anderi Sharf).

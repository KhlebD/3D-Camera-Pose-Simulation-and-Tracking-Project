#include <filesystem>
namespace fs = std::filesystem;

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <limits>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb/stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "Texture.h"
#include "shaderClass.h"
#include "VAO.h"
#include "VBO.h"
#include "EBO.h"
#include "Camera.h"

const unsigned int width = 800;
const unsigned int height = 800;
const int TERRAIN_SIZE = 50; // Size of the grid
const float TERRAIN_SCALE = 0.2f; // Scale of the terrain

float fovX = 45.0f; // Example FOV in degrees
float aspectRatio = static_cast<float>(width) / height;
double fx = (width / 2.0) / tan(glm::radians(fovX / 2.0));
double fy = fx; // Assuming square pixels
double cx = width / 2.0; // Center of the image
double cy = height / 2.0; // Center of the image

cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // Assuming no lens distortion

enum Mode { NORMAL, TRACKER_SETUP };

Mode currentMode = NORMAL;

struct CameraPose {
    glm::vec3 position;
    glm::vec3 orientation;
};

struct OpenGLContext {
    GLFWwindow* window;
    Shader* shaderProgram;
    VAO* VAO1;
    VBO* VBO1;
    EBO* EBO1;
    Texture* brickTex;
    Camera* camera;
    Shader* pickedPointsShader;
    VAO* pickedPointsVAO;
    VBO* pickedPointsVBO;
    VBO* colorVBO;
    Shader* screenShader;
    GLuint screenTexture;
};

struct TrackingAndRecordingSystem {
    std::vector<glm::vec3> pickedPoints3D;
    std::vector<glm::vec3> pickedColors;
    std::vector<CameraPose> trueCameraPoses;
    std::vector<CameraPose> computedCameraPoses;
};

// Global Variables:
TrackingAndRecordingSystem tars;
bool movedSinceLastB = false;
CameraPose lastCtrlPlusEPose;
volatile bool estimatedCameraViewShouldBeActive = false;
int currentCameraIndex = -1;
CameraPose globalViewPose = {
    glm::vec3(20.0f, 20.0f, 20.0f),  // position
    glm::vec3(-0.5f, -0.4f, -0.5f)   // orientation
};

glm::vec4 backgroundColor = glm::vec4(0.1f, 0.3f, 0.5f, 1.0f); // Global variable for background color

struct CallbackData {
    OpenGLContext* context;
};

struct Triangle {
    glm::vec3 vertices[3];
    glm::vec3 color;
};

void createCameraTriangles(const CameraPose& pose, std::vector<Triangle>& triangles, glm::vec3 color) {
    glm::vec3 tip = pose.position;
    glm::vec3 dir = glm::normalize(pose.orientation);
    
    // Calculate the direction from the camera to the global view
    glm::vec3 toGlobal = glm::normalize(globalViewPose.position - pose.position);
    
    // Calculate the perpendicular vector
    glm::vec3 perpendicular = glm::cross(dir, toGlobal);
    
    // Check if the perpendicular vector is zero (or very close to zero)
    if (glm::length(perpendicular) < 0.001f) {
        // If so, choose an arbitrary perpendicular vector
        perpendicular = glm::normalize(glm::cross(dir, glm::vec3(0.0f, 1.0f, 0.0f)));
    } else {
        perpendicular = glm::normalize(perpendicular);
    }

    // Calculate the midpoint of the base
    glm::vec3 midBase = tip + dir * 1.0f; // Adjust the length as needed

    // Calculate the base vertices
    glm::vec3 base1 = midBase + perpendicular * 0.1f; // Adjust the width as needed
    glm::vec3 base2 = midBase - perpendicular * 0.1f;

    Triangle triangle;
    triangle.vertices[0] = tip;
    triangle.vertices[1] = base1;
    triangle.vertices[2] = base2;
    triangle.color = color;

    triangles.push_back(triangle);
}

void renderCameraTriangles(OpenGLContext& context) {
    std::vector<Triangle> cameraTriangles;

    for (size_t i = 0; i < tars.trueCameraPoses.size() && i < tars.computedCameraPoses.size(); ++i) {
        // Blue triangle for true pose
        createCameraTriangles(tars.trueCameraPoses[i], cameraTriangles, glm::vec3(0.0f, 0.0f, 1.0f));
        
        // Red triangle for computed pose
        createCameraTriangles(tars.computedCameraPoses[i], cameraTriangles, glm::vec3(1.0f, 0.0f, 0.0f));
    }

    if (!cameraTriangles.empty()) {
        if (!movedSinceLastB) {
            int index = currentCameraIndex * 2; 
            cameraTriangles[index].color = glm::vec3(1.0f) - cameraTriangles[index].color;

            cameraTriangles[index + 1].color = glm::vec3(1.0f) - cameraTriangles[index + 1].color;
        }

        std::vector<GLfloat> vertexData;
        for (const auto& triangle : cameraTriangles) {
            for (const auto& vertex : triangle.vertices) {
                vertexData.insert(vertexData.end(), {vertex.x, vertex.y, vertex.z, triangle.color.r, triangle.color.g, triangle.color.b});
            }
        }

        GLuint VAO, VBO;
        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(GLfloat), vertexData.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glDrawArrays(GL_TRIANGLES, 0, vertexData.size() / 6);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        glDeleteBuffers(1, &VBO);
        glDeleteVertexArrays(1, &VAO);
    }
}

float generateRandomFloat(float lower, float upper) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(lower, upper);
    return dis(gen);
}

float colorDistance(const glm::vec3& c1, const glm::vec3& c2) {
    return glm::distance(c1, c2);
}

glm::vec3 generateUniqueColor(const std::vector<glm::vec3>& existingColors, float threshold = 0.1f) {
    glm::vec3 newColor;
    bool unique;
    do {
        newColor = glm::vec3(generateRandomFloat(0.0f, 1.0f), generateRandomFloat(0.0f, 1.0f), generateRandomFloat(0.0f, 1.0f));
        unique = true;
        for (const auto& color : existingColors) {
            if (colorDistance(newColor, color) < threshold) {
                unique = false;
                break;
            }
        }
    } while (!unique);
    return newColor;
}

float mountainHeight(float x, float z, float radius, float peakHeight) {
    float distanceFromCenter = std::sqrt(x*x + z*z);
    if (distanceFromCenter > radius) {
        return 0.0f;
    }

    float innerRadius = radius / 3.0f;
    float distanceFromCircumference = radius - distanceFromCenter;
    float normalizedDistance = distanceFromCircumference / radius;
    float baseHeight = peakHeight * normalizedDistance * normalizedDistance;

    if (distanceFromCenter < innerRadius) {
        // Calculate height at inner radius
        float innerHeight = mountainHeight(innerRadius, 0, radius, peakHeight);
        
        // Add the squared distance from center
        float distanceFromInnerCircumference = innerRadius - distanceFromCenter;
        float additionalHeight = distanceFromInnerCircumference * distanceFromInnerCircumference;
        
        return innerHeight + additionalHeight;
    }

    return baseHeight;
}

std::vector<GLfloat> createTerrainVertices() {
    std::vector<GLfloat> vertices;
    float mountainRadius = 2.0f;
    float mountainPeakHeight = 4.0f;

    for (int z = 0; z < TERRAIN_SIZE; z++) {
        for (int x = 0; x < TERRAIN_SIZE; x++) {
            float xPos = (x - TERRAIN_SIZE / 2) * TERRAIN_SCALE;
            float zPos = (z - TERRAIN_SIZE / 2) * TERRAIN_SCALE;
            float baseHeight = sin(xPos * 1.5f) * cos(zPos * 1.5f) * 0.2f;
            
            // Add mountain height
            float mountainAddition = mountainHeight(xPos, zPos, mountainRadius, mountainPeakHeight);
            float height = baseHeight + mountainAddition;

            // Position
            vertices.push_back(xPos);
            vertices.push_back(height);
            vertices.push_back(zPos);

            // Color (adjusted based on height for visual effect)
            float colorIntensity = glm::clamp(height / 4.0f, 0.0f, 1.0f);
            vertices.push_back(0.2f + colorIntensity * 0.3f);
            vertices.push_back(0.4f + colorIntensity * 0.3f);
            vertices.push_back(0.1f + colorIntensity * 0.3f);

            // Texture coordinates
            vertices.push_back(static_cast<float>(x) / TERRAIN_SIZE);
            vertices.push_back(static_cast<float>(z) / TERRAIN_SIZE);
        }
    }
    return vertices;
}

std::vector<GLuint> createTerrainIndices() {
    std::vector<GLuint> indices;
    for (int z = 0; z < TERRAIN_SIZE - 1; z++) {
        for (int x = 0; x < TERRAIN_SIZE - 1; x++) {
            int topLeft = z * TERRAIN_SIZE + x;
            int topRight = topLeft + 1;
            int bottomLeft = (z + 1) * TERRAIN_SIZE + x;
            int bottomRight = bottomLeft + 1;

            // First triangle
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // Second triangle
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
    return indices;
}

std::vector<GLfloat> terrainVertices = createTerrainVertices();
std::vector<GLuint> terrainIndices = createTerrainIndices();

bool savePickedPoints(const OpenGLContext& context, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return false;
    }

    size_t size = tars.pickedPoints3D.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (size_t i = 0; i < size; ++i) {
        file.write(reinterpret_cast<const char*>(&tars.pickedPoints3D[i]), sizeof(glm::vec3));
        file.write(reinterpret_cast<const char*>(&tars.pickedColors[i]), sizeof(glm::vec3));
    }

    return true;
}

void loadPickedPoints(OpenGLContext& context, const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return;

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    tars.pickedPoints3D.resize(size);
    tars.pickedColors.resize(size);

    for (size_t i = 0; i < size; ++i) {
        file.read(reinterpret_cast<char*>(&tars.pickedPoints3D[i]), sizeof(glm::vec3));
        file.read(reinterpret_cast<char*>(&tars.pickedColors[i]), sizeof(glm::vec3));
    }
}

OpenGLContext createContext(int width, int height, const char* title, CameraPose cameraPose) {
    OpenGLContext context;

    context.window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (context.window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(context.window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        exit(-1);
    }

    glViewport(0, 0, width, height);

    context.shaderProgram = new Shader("default.vert", "default.frag");
    context.VAO1 = new VAO();
    context.VAO1->Bind();
    context.VBO1 = new VBO(terrainVertices.data(), terrainVertices.size() * sizeof(GLfloat));
    context.EBO1 = new EBO(terrainIndices.data(), terrainIndices.size() * sizeof(GLuint));

    context.VAO1->LinkAttrib(*context.VBO1, 0, 3, GL_FLOAT, 8 * sizeof(float), (void*)0);
    context.VAO1->LinkAttrib(*context.VBO1, 1, 3, GL_FLOAT, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    context.VAO1->LinkAttrib(*context.VBO1, 2, 2, GL_FLOAT, 8 * sizeof(float), (void*)(6 * sizeof(float)));

    context.VAO1->Unbind();
    context.VBO1->Unbind();
    context.EBO1->Unbind();

    std::string parentDir = (fs::current_path()).string();
    std::string texPath = "/Resources/Textures/";

    context.brickTex = new Texture((parentDir + texPath + "tex.jpg").c_str(), GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
    context.brickTex->texUnit(*context.shaderProgram, "tex0", 0);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    context.camera = new Camera(width, height, cameraPose.position);
    context.camera->Orientation = cameraPose.orientation;

    context.pickedPointsShader = new Shader("point.vert", "point.frag");

    context.pickedPointsVAO = new VAO();
    context.pickedPointsVAO->Bind();

    context.pickedPointsVBO = new VBO(nullptr, sizeof(glm::vec3) * 1000);
    context.colorVBO = new VBO(nullptr, sizeof(glm::vec3) * 1000);
    context.pickedPointsVAO->LinkAttrib(*context.pickedPointsVBO, 0, 3, GL_FLOAT, sizeof(glm::vec3), (void*)0);
    context.pickedPointsVAO->LinkAttrib(*context.colorVBO, 1, 3, GL_FLOAT, sizeof(glm::vec3), (void*)0);

    context.pickedPointsVAO->Unbind();
    context.pickedPointsVBO->Unbind();
    context.colorVBO->Unbind();

    loadPickedPoints(context, "picked_points.dat");

    context.screenShader = new Shader("screen.vert", "screen.frag");

    // Create a texture to store the blended image
    glGenTextures(1, &context.screenTexture);
    glBindTexture(GL_TEXTURE_2D, context.screenTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return context;
}

void renderContext(OpenGLContext& context) {
    glfwMakeContextCurrent(context.window);

    glClearColor(backgroundColor.r, backgroundColor.g, backgroundColor.b, backgroundColor.a); // Use global background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    context.shaderProgram->Activate();
    context.camera->Matrix(45.0f, 0.1f, 100.0f, *context.shaderProgram, "camMatrix");

    context.brickTex->Bind();
    context.VAO1->Bind();
    glDrawElements(GL_TRIANGLES, terrainIndices.size(), GL_UNSIGNED_INT, 0);
}

void deleteContext(OpenGLContext& context) {
    delete context.VAO1;
    delete context.VBO1;
    delete context.EBO1;
    delete context.brickTex;
    delete context.shaderProgram;
    delete context.camera;
    delete context.pickedPointsShader;
    delete context.pickedPointsVAO;
    delete context.pickedPointsVBO;
    delete context.colorVBO;
    glDeleteTextures(1, &context.screenTexture);
    glfwDestroyWindow(context.window);
}

glm::vec3 rayTerrainIntersection(const glm::vec3& rayOrigin, const glm::vec3& rayDirection) {
    glm::vec3 intersection = rayOrigin;
    float stepSize = 0.1f; // Step size for ray marching

    float mountainRadius = 2.0f;
    float mountainPeakHeight = 4.0f;

    for (int i = 0; i < 1000; ++i) { // Max 1000 iterations
        intersection += stepSize * rayDirection;

        // Clamp intersection within terrain bounds
        float x = glm::clamp(intersection.x, -TERRAIN_SIZE * TERRAIN_SCALE / 2.0f, TERRAIN_SIZE * TERRAIN_SCALE / 2.0f);
        float z = glm::clamp(intersection.z, -TERRAIN_SIZE * TERRAIN_SCALE / 2.0f, TERRAIN_SIZE * TERRAIN_SCALE / 2.0f);

        // Calculate the base height of the terrain at this (x, z) position
        float baseHeight = sin(x * 1.5f) * cos(z * 1.5f) * 0.2f;

        // Add mountain height
        float mountainAddition = mountainHeight(x, z, mountainRadius, mountainPeakHeight);
        float terrainHeight = baseHeight + mountainAddition;

        if (intersection.y <= terrainHeight) {
            intersection.y = terrainHeight;
            break;
        }
    }

    return intersection;
}

cv::Mat captureFramebuffer(int width, int height) {
    cv::Mat image(height, width, CV_8UC3);
    glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, image.data);
    cv::flip(image, image, 0);
    return image;
}

std::vector<cv::Point> findColorPixels(const cv::Mat& image, const glm::vec3& targetColor, float threshold = 30.0f) {
    std::vector<cv::Point> points;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            glm::vec3 color(pixel[2], pixel[1], pixel[0]);
            if (glm::distance(color, targetColor) < threshold) {
                points.emplace_back(x, y);
            }
        }
    }
    return points;
}

glm::vec2 computeCentroid(const std::vector<cv::Point>& points) {
    glm::vec2 centroid(0.0f, 0.0f);
    for (const auto& point : points) {
        centroid += glm::vec2(point.x, point.y);
    }
    centroid /= static_cast<float>(points.size());

    return centroid;
}

bool performPnP(const std::vector<cv::Point2f>& imagePoints, const std::vector<cv::Point3f>& objectPoints, cv::Mat& cameraPosition, cv::Mat& cameraOrientation) {
    if (imagePoints.size() < 6) {
        if (imagePoints.size() == 0) {
            std::cout << "No trackers are visible in the main view. "
                      << "\nAt least 6 trackers should be visible in the main view for our PnP algorithm to be used.\n" << std::endl;
        } else {
            std::cout << "Only " << (imagePoints.size() == 1 ? "one tracker is" : imagePoints.size() == 2 ? "two trackers are" : imagePoints.size() == 3 ? "three trackers are" : imagePoints.size() == 4 ? "four trackers are" : "five trackers are")
                      << " visible in the main view. "
                      << "\nAt least 6 trackers should be visible in the main view for our PnP algorithm to be used.\n" << std::endl;
        }
        return false;
    }
    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

    if (success) {
        cv::Rodrigues(rvec, cameraOrientation);
        cameraPosition = -cameraOrientation.t() * tvec;
    }

    return success;
}

void findPoints(OpenGLContext& context, std::vector<cv::Point2f>& imagePoints, std::vector<cv::Point3f>& objectPoints) {
    cv::Mat image = captureFramebuffer(width, height); 

    for (size_t i = 0; i < tars.pickedColors.size(); ++i) {
        glm::vec3 targetColor = tars.pickedColors[i] * 255.0f; 
        std::vector<cv::Point> points = findColorPixels(image, targetColor); 
        if (!points.empty()) { 
            glm::vec2 centroid = computeCentroid(points);  

            imagePoints.push_back(cv::Point2f(centroid.x, centroid.y));
            objectPoints.push_back(cv::Point3f(tars.pickedPoints3D[i].x, tars.pickedPoints3D[i].y, tars.pickedPoints3D[i].z));
        }
    }
}

void printCameraData(const CameraPose& truePose, const CameraPose& computedPose) {
    std::cout << "True Camera Position: (" << truePose.position.x << ", "
              << truePose.position.y << ", " << truePose.position.z << ")\n"
              << "True Camera Orientation: (" << truePose.orientation.x << ", "
              << truePose.orientation.y << ", " << truePose.orientation.z << ")\n"
              << "Computed Camera Position: (" << computedPose.position.x << ", "
              << computedPose.position.y << ", " << computedPose.position.z << ")\n"
              << "Computed Camera Orientation: (" << computedPose.orientation.x << ", "
              << computedPose.orientation.y << ", " << computedPose.orientation.z << ")\n" << std::endl;
}

void updateCameraView(OpenGLContext& context) {
    if (currentCameraIndex >= 0 && currentCameraIndex < tars.trueCameraPoses.size()) {
        context.camera->Position = tars.trueCameraPoses[currentCameraIndex].position;
        context.camera->Orientation = tars.trueCameraPoses[currentCameraIndex].orientation;
    }
}

void processCameraTracking(OpenGLContext& context, bool record) {
    glfwMakeContextCurrent(context.window);

    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;

    findPoints(context, imagePoints, objectPoints);

    cv::Mat estimatedPosition, estimatedOrientation;
    if (performPnP(imagePoints, objectPoints, estimatedPosition, estimatedOrientation)) {
        glm::vec3 actualPosition = context.camera->Position;
        glm::vec3 actualDirection = glm::normalize(context.camera->Orientation);
        glm::vec3 estimatedDir(estimatedOrientation.at<double>(2, 0), estimatedOrientation.at<double>(2, 1), estimatedOrientation.at<double>(2, 2));
        if(record) { // B key pressed
            tars.trueCameraPoses.push_back({ actualPosition, actualDirection });
            tars.computedCameraPoses.push_back({ glm::vec3(estimatedPosition.at<double>(0), estimatedPosition.at<double>(1), estimatedPosition.at<double>(2)), estimatedDir });
        }

        else { // right mouse click
            lastCtrlPlusEPose.position = glm::vec3(estimatedPosition.at<double>(0), estimatedPosition.at<double>(1), estimatedPosition.at<double>(2));
            lastCtrlPlusEPose.orientation = estimatedDir;
        }

        printCameraData({ actualPosition, actualDirection }, { glm::vec3(estimatedPosition.at<double>(0), estimatedPosition.at<double>(1), estimatedPosition.at<double>(2)), estimatedDir });
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    CallbackData* data = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
    OpenGLContext* context = data->context;

    if (currentMode == TRACKER_SETUP) {
        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
            double mouseX, mouseY;
            glfwGetCursorPos(window, &mouseX, &mouseY);

            float x = (2.0f * mouseX) / (float)width - 1.0f;
            float y = 1.0f - (2.0f * mouseY) / (float)height;

            glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
            glm::mat4 view = context->camera->GetViewMatrix();
            glm::mat4 invVP = glm::inverse(projection * view);

            glm::vec4 rayStart = invVP * glm::vec4(x, y, -1.0, 1.0);
            glm::vec4 rayEnd = invVP * glm::vec4(x, y, 1.0, 1.0);
            rayStart /= rayStart.w;
            rayEnd /= rayEnd.w;

            glm::vec3 rayDir = glm::normalize(glm::vec3(rayEnd - rayStart));
            glm::vec3 rayOrigin = glm::vec3(rayStart);

            glm::vec3 intersection = rayTerrainIntersection(rayOrigin, rayDir);

            if (mods & GLFW_MOD_CONTROL) {
                if (!tars.pickedPoints3D.empty()) {
                    float minDistance = std::numeric_limits<float>::max();
                    size_t closestIndex = 0;
                    for (size_t i = 0; i < tars.pickedPoints3D.size(); ++i) {
                        float distance = glm::distance(intersection, tars.pickedPoints3D[i]);
                        if (distance < minDistance) {
                            minDistance = distance;
                            closestIndex = i;
                        }
                    }

                    glm::vec3 deletedPoint = tars.pickedPoints3D[closestIndex];
                    tars.pickedPoints3D.erase(tars.pickedPoints3D.begin() + closestIndex);
                    tars.pickedColors.erase(tars.pickedColors.begin() + closestIndex);

                    std::cout << "Deleted point: (" << deletedPoint.x << ", " << deletedPoint.y << ", " << deletedPoint.z << ")" << std::endl;
                }
            } else {
                std::vector<glm::vec3> existingColors = tars.pickedColors;
                existingColors.emplace_back(0.1f, 0.3f, 0.5f);
                tars.pickedPoints3D.push_back(intersection);
                tars.pickedColors.push_back(generateUniqueColor(existingColors));

                std::cout << "Picked point: (" << intersection.x << ", " << intersection.y << ", " << intersection.z << ")" << std::endl;
            }
        }
    }
    
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    CallbackData* data = static_cast<CallbackData*>(glfwGetWindowUserPointer(window));
    OpenGLContext* context = data->context;

    if (currentMode == TRACKER_SETUP && key == GLFW_KEY_S && mods == GLFW_MOD_CONTROL && action == GLFW_PRESS) {
        bool success = savePickedPoints(*context, "picked_points.dat");
        if (success) {
            std::cout << "Picked points saved successfully." << std::endl;
        } else {
            std::cout << "Failed to save picked points." << std::endl;
        }
    }

    if (currentMode == NORMAL && key == GLFW_KEY_E && mods == GLFW_MOD_CONTROL && action == GLFW_PRESS) {
        if(!estimatedCameraViewShouldBeActive) {
            processCameraTracking(*context, false);
        }  
        estimatedCameraViewShouldBeActive = !estimatedCameraViewShouldBeActive;
    }

    if (key == GLFW_KEY_M && mods == GLFW_MOD_CONTROL && action == GLFW_PRESS) {
        if (currentMode == NORMAL) {
            currentMode = TRACKER_SETUP;
            std::cout << "\n-------------------------------------------\nSwitched to Tracker Setup Mode.\n" << std::endl;
        } else {
            currentMode = NORMAL;
            std::cout << "\n-------------------------------------------\nSwitched to Normal Mode.\n" << std::endl;
        }
    }

    if (currentMode == NORMAL && key == GLFW_KEY_B && action == GLFW_PRESS) {
        processCameraTracking(*context, true);
        currentCameraIndex = tars.trueCameraPoses.size() - 1;
        movedSinceLastB = false;
    }

    if (currentMode == NORMAL && key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
        if (!tars.trueCameraPoses.empty()) {
            bool updateFlag = true;
            if (currentCameraIndex == 0 && !movedSinceLastB) {
                updateFlag = false;
            }
            if (currentCameraIndex > 0) {
                currentCameraIndex--;
            }
            if (updateFlag) {
                updateCameraView(*context);
                printCameraData(tars.trueCameraPoses[currentCameraIndex], tars.computedCameraPoses[currentCameraIndex]);
                movedSinceLastB = false;
            }
        }
    }

    if (currentMode == NORMAL && key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
        if (!tars.trueCameraPoses.empty()) {
            bool updateFlag = true;
            if (currentCameraIndex == tars.trueCameraPoses.size() - 1 && !movedSinceLastB) {
                updateFlag = false;
            }
            if (currentCameraIndex < tars.trueCameraPoses.size() - 1) {
                currentCameraIndex++;
            }
            if (updateFlag) {
                updateCameraView(*context);
                printCameraData(tars.trueCameraPoses[currentCameraIndex], tars.computedCameraPoses[currentCameraIndex]);
                movedSinceLastB = false;
            }
        }
    }
}

cv::Mat blendImages(const cv::Mat& img1, const cv::Mat& img2, float alpha) {
    cv::Mat blended = cv::Mat::zeros(img1.size(), img1.type());

    for (int y = 0; y < img1.rows; ++y) {
        for (int x = 0; x < img1.cols; ++x) {
            cv::Vec3b color1 = img1.at<cv::Vec3b>(y, x);
            cv::Vec3b color2 = img2.at<cv::Vec3b>(y, x);
            cv::Vec3b blendedColor;

            glm::vec3 pixelColor(color2[2], color2[1], color2[0]);
            glm::vec3 globalBgColor(0.0f, 0.0f, 0.0f);

            if (pixelColor == globalBgColor) {
                blendedColor = color1; // Use the pixel from the first image
            } else {
                for (int c = 0; c < 3; ++c) {
                    blendedColor[c] = static_cast<uchar>(alpha * color1[c] + (1 - alpha) * color2[c]);
                }
            }

            blended.at<cv::Vec3b>(y, x) = blendedColor;
        }
    }

    return blended;
}


void renderPickedPoints(OpenGLContext& context) {
    if (tars.pickedPoints3D.empty()) return;
    context.pickedPointsShader->Activate();

    glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    glm::mat4 view = context.camera->GetViewMatrix();

    glUniformMatrix4fv(glGetUniformLocation(context.pickedPointsShader->ID, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniformMatrix4fv(glGetUniformLocation(context.pickedPointsShader->ID, "view"), 1, GL_FALSE, glm::value_ptr(view));

    context.pickedPointsVAO->Bind();

    glBindBuffer(GL_ARRAY_BUFFER, context.pickedPointsVBO->ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, tars.pickedPoints3D.size() * sizeof(glm::vec3), tars.pickedPoints3D.data());

    glBindBuffer(GL_ARRAY_BUFFER, context.colorVBO->ID);
    glBufferSubData(GL_ARRAY_BUFFER, 0, tars.pickedColors.size() * sizeof(glm::vec3), tars.pickedColors.data());

    glPointSize(10.0f);
    glEnable(GL_PROGRAM_POINT_SIZE);

    glDrawArrays(GL_POINTS, 0, tars.pickedPoints3D.size());

    context.pickedPointsVAO->Unbind();
}

void trackerSetupMode(GLFWwindow* mainWindow) {
    CameraPose cameraPose;
    cameraPose.position = glm::vec3(6.0f, 20.0f, 6.0f);
    cameraPose.orientation = glm::vec3(-0.5f, -1.7f, -0.5f);
    OpenGLContext context = createContext(width, height, "Tracker Setup", cameraPose);
    CallbackData callbackData;
    callbackData.context = &context;
    glfwSetWindowUserPointer(context.window, &callbackData);
    glfwSetMouseButtonCallback(context.window, mouseButtonCallback);
    glfwSetKeyCallback(context.window, keyCallback);

    while (!glfwWindowShouldClose(context.window)) {
        context.camera->Inputs(context.window);

        glfwMakeContextCurrent(context.window);
        renderContext(context);
        renderPickedPoints(context);
        glfwSwapBuffers(context.window);

        glfwPollEvents();

        if (currentMode == NORMAL) {
            break;
        }
    }

    deleteContext(context);
}

void renderTexture(GLuint textureID, int width, int height, Shader& shader) {
    static GLuint VAO = 0, VBO = 0, EBO = 0;
    if (VAO == 0) {
        float vertices[] = {
            // positions   // texCoords
            -1.0f,  1.0f,  1.0f, 1.0f, // Top-left vertex (rotated 180 degrees)
            -1.0f, -1.0f,  1.0f, 0.0f, // Bottom-left vertex (rotated 180 degrees)
             1.0f, -1.0f,  0.0f, 0.0f, // Bottom-right vertex (rotated 180 degrees)
             1.0f,  1.0f,  0.0f, 1.0f  // Top-right vertex (rotated 180 degrees)
        };
        GLuint indices[] = {
            0, 1, 2,
            0, 2, 3
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    shader.Activate();
    glBindVertexArray(VAO);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    CameraPose explorerViewStartingPose;
    explorerViewStartingPose.position = glm::vec3(10.0f, 10.0f, 10.0f);
    explorerViewStartingPose.orientation = glm::vec3(-0.5f, -0.5f, -0.5f);

    OpenGLContext explorerContext = createContext(width, height, "Explorer View", explorerViewStartingPose);
    OpenGLContext globalContext = createContext(width, height, "Global View", globalViewPose);

    CallbackData callbackData;
    callbackData.context = &explorerContext;

    glfwSetWindowUserPointer(explorerContext.window, &callbackData);
    glfwSetMouseButtonCallback(explorerContext.window, mouseButtonCallback);
    glfwSetKeyCallback(explorerContext.window, keyCallback);

    while (!glfwWindowShouldClose(explorerContext.window) && !glfwWindowShouldClose(globalContext.window)) {
        glfwMakeContextCurrent(explorerContext.window);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cv::Mat displayImage;

        explorerContext.shaderProgram->Activate();

        // Render explorer view
        glDisable(GL_BLEND);
        glUniform1i(glGetUniformLocation(explorerContext.shaderProgram->ID, "applyFilter"), GL_FALSE);
        renderContext(explorerContext);

        if (estimatedCameraViewShouldBeActive) {
            // Capture the framebuffer of the explorer view
            cv::Mat explorerImage = captureFramebuffer(width, height);

            // Store original camera position and orientation
            glm::vec3 originalPosition = explorerContext.camera->Position;
            glm::vec3 originalOrientation = explorerContext.camera->Orientation;

            // Switch to estimated camera position and orientation
            explorerContext.camera->Position = lastCtrlPlusEPose.position;
            explorerContext.camera->Orientation = lastCtrlPlusEPose.orientation;

            // Save the original background color
            glm::vec4 originalBackgroundColor = backgroundColor;
            // Set the background color to black
            glm::vec4 blackColor = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            backgroundColor = blackColor;

            // Render and capture the framebuffer of the estimated view
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glUniform3f(glGetUniformLocation(explorerContext.shaderProgram->ID, "filterColor"), 0.5f, 0.0f, 0.5f);
            glUniform1f(glGetUniformLocation(explorerContext.shaderProgram->ID, "filterAlpha"), 0.3f);
            glUniform1i(glGetUniformLocation(explorerContext.shaderProgram->ID, "applyFilter"), GL_TRUE);
            renderContext(explorerContext);
            cv::Mat estimatedImage = captureFramebuffer(width, height);

            // Reset the background color to the original
            backgroundColor = originalBackgroundColor;

            // Reset to original camera position and orientation
            explorerContext.camera->Position = originalPosition;
            explorerContext.camera->Orientation = originalOrientation;

            // Blend the images using the global background color
            displayImage = blendImages(explorerImage, estimatedImage, 0.5f); // 50% blending

            // Rotate the image by 180 degrees (vertical and horizontal flip)
            cv::flip(displayImage, displayImage, -1); // Both vertical and horizontal flip

            // Update the texture with the rotated image
            glBindTexture(GL_TEXTURE_2D, explorerContext.screenTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, displayImage.cols, displayImage.rows, GL_BGR, GL_UNSIGNED_BYTE, displayImage.data);

            // Render the texture to the screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            renderTexture(explorerContext.screenTexture, width, height, *explorerContext.screenShader);

        } else {
            renderPickedPoints(explorerContext);    
        }

        // Render picked points on top of everything
        glfwSwapBuffers(explorerContext.window);

        if (explorerContext.camera->Inputs(explorerContext.window)) {
            movedSinceLastB = true;
            estimatedCameraViewShouldBeActive = false;
        }

        // Render Global View (globalContext)
        glfwMakeContextCurrent(globalContext.window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        globalContext.shaderProgram->Activate();
        glUniform1i(glGetUniformLocation(globalContext.shaderProgram->ID, "applyFilter"), GL_FALSE);
        renderContext(globalContext);
        renderPickedPoints(globalContext);
        renderCameraTriangles(globalContext);
        glfwSwapBuffers(globalContext.window);

        glfwPollEvents();

        if (currentMode == TRACKER_SETUP) {
            glfwHideWindow(explorerContext.window);
            glfwHideWindow(globalContext.window);
            trackerSetupMode(explorerContext.window);

            glfwShowWindow(globalContext.window);
            glfwShowWindow(explorerContext.window);
            
            tars.trueCameraPoses = std::vector<CameraPose>();
            tars.computedCameraPoses = std::vector<CameraPose>();
            int currentCameraIndex = -1;
            explorerContext.camera = new Camera(width, height, explorerViewStartingPose.position);
            explorerContext.camera->Orientation = explorerViewStartingPose.orientation;
            currentMode = NORMAL;
        }
    }

    deleteContext(explorerContext);
    deleteContext(globalContext);
    glfwTerminate();
    return 0;
}

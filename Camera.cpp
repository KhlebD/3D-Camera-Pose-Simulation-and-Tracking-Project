#include "Camera.h"

Camera::Camera(int width, int height, glm::vec3 position)
{
    Camera::width = width;
    Camera::height = height;
    Position = position;
}

void Camera::Matrix(float FOVdeg, float nearPlane, float farPlane, Shader& shader, const char* uniform)
{
    // Initializes matrices since otherwise they will be the null matrix
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    // Makes camera look in the right direction from the right position
    view = glm::lookAt(Position, Position + Orientation, Up);
    // Adds perspective to the scene
    projection = glm::perspective(glm::radians(FOVdeg), (float)width / height, nearPlane, farPlane);

    // Exports the camera matrix to the Vertex Shader
    glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(projection * view));
}

glm::mat4 Camera::GetViewMatrix()
{
    return glm::lookAt(Position, Position + Orientation, Up);
}

bool Camera::isCtrlPressed(GLFWwindow* window)
{
    return glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
}

bool Camera::Inputs(GLFWwindow* window) {
    // Store previous position and orientation
    glm::vec3 prevPosition = Position;
    glm::vec3 prevOrientation = Orientation;

    // Prevent camera movement if Alt key is pressed
    if (isCtrlPressed(window)) return false;

    // Toggle speed modes on alt press
    static bool altPressed = false;
    if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS && !altPressed) {
        altPressed = true;
        currentSpeedMode = static_cast<SpeedMode>((currentSpeedMode + 1) % 3);
    } else if (glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_RELEASE) {
        altPressed = false;
    }

    // Set speed based on current mode
    float currentSpeed = speed;
    if (currentSpeedMode == SLOW) {
        currentSpeed *= 0.2f; // Five times slower
    } else if (currentSpeedMode == VERY_SLOW) {
        currentSpeed *= 0.05f; // Twenty times slower
    }

    // Handles key inputs
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        Position += currentSpeed * Orientation;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        Position += currentSpeed * -glm::normalize(glm::cross(Orientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        Position += currentSpeed * -Orientation;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        Position += currentSpeed * glm::normalize(glm::cross(Orientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        Position += currentSpeed * Up;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        Position += currentSpeed * -Up;
    }

    // Handles mouse inputs
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

        if (firstClick) {
            glfwSetCursorPos(window, (width / 2), (height / 2));
            firstClick = false;
        }

        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        float rotX = sensitivity * (float)(mouseY - (height / 2)) / height;
        float rotY = sensitivity * (float)(mouseX - (width / 2)) / width;

        glm::vec3 newOrientation = glm::rotate(Orientation, glm::radians(-rotX), glm::normalize(glm::cross(Orientation, Up)));

        if (abs(glm::angle(newOrientation, Up) - glm::radians(90.0f)) <= glm::radians(85.0f)) {
            Orientation = newOrientation;
        }

        Orientation = glm::rotate(Orientation, glm::radians(-rotY), Up);

        glfwSetCursorPos(window, (width / 2), (height / 2));
    } else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstClick = true;
    }

    // Check if the camera moved
    return (Position != prevPosition || Orientation != prevOrientation);
}


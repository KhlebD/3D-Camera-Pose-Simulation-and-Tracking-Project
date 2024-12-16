#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include <glad/glad.h>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cerrno>
#include <glm/glm.hpp>

std::string get_file_contents(const char* filename);

class Shader
{
public:
    // Reference ID of the Shader Program
    GLuint ID;
    // Constructor that builds the Shader Program from 2 different shaders
    Shader(const char* vertexFile, const char* fragmentFile);

    // Activates the Shader Program
    void Activate();
    // Sets a vec3 uniform
    void setVec3(const std::string& name, const glm::vec3& value) const;
    // Sets a boolean uniform
    void setBool(const std::string& name, bool value) const;
    // Deletes the Shader Program
    void Delete();

private:
    // Checks if the different Shaders have compiled properly
    void compileErrors(unsigned int shader, const char* type);
};

#endif

#version 330 core

layout(location = 0) in vec3 aPos; // Position attribute
layout(location = 1) in vec3 aColor; // Color attribute

out vec3 fragColor; // Pass color to fragment shader

uniform mat4 projection;
uniform mat4 view;

void main()
{
    fragColor = aColor; // Set the fragment color
    gl_Position = projection * view * vec4(aPos, 1.0);
}

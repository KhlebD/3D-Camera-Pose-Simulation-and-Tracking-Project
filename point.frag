#version 330 core

out vec4 FragColor;

in vec3 fragColor;

uniform bool applyFilter;
uniform vec3 filterColor;
uniform float filterAlpha; // New uniform for controlling filter strength

void main()
{
    vec3 finalColor = fragColor;
    float alpha = 1.0; // Default to fully opaque

    if (applyFilter) {
        finalColor = mix(fragColor, filterColor, filterAlpha);
        alpha = 0.7; // Adjust this value to control overall translucency when filter is applied
    }

    FragColor = vec4(finalColor, alpha);
}
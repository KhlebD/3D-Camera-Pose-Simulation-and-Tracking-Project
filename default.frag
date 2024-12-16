#version 330 core

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D tex0;
uniform vec3 filterColor;
uniform float filterAlpha;
uniform bool applyFilter;

void main() {
    vec4 texColor = texture(tex0, texCoord);
    
    if (applyFilter) {
        vec3 finalColor = mix(texColor.rgb, filterColor, filterAlpha);
        FragColor = vec4(finalColor, 1.0); // Keep alpha at 1.0 to blend with existing scene
    } else {
        FragColor = texColor;
    }
}
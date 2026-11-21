#version 450
layout(set = 0, binding = 1, std430) readonly buffer SelectedColor {
    vec4 color;
} selectedColor;
layout(location = 0) out vec4 outputColor;

void main() {
    outputColor = selectedColor.color;
}

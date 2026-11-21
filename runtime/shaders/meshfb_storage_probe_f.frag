#version 450
// Storage in the actual final graphics shader, recorded only when that phase is issued.
layout(location = 0) in vec4 i_color;
layout(location = 0) out vec4 o_color;
layout(set = 0, binding = 1, r32ui) uniform uimage2D final_result;
void main() {
    imageAtomicOr(final_result, ivec2(0, 0), 0x01000000u);
    o_color = i_color;
}

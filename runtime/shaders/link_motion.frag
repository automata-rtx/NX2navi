#version 450
layout(location = 15) in vec4 previous;
layout(binding = 1, std140) uniform Motion {
    vec4 viewport;
    vec4 history;
};
layout(binding = 2) uniform sampler2D previousDistance;
layout(location = 0) out vec2 motion;
void main() {
    if (!(previous.w > 1e-6)) discard;
    vec2 previousWindow = (previous.xy / previous.w * vec2(0.5, -0.5) + 0.5) * viewport.xy;
    motion = previousWindow - (gl_FragCoord.xy - viewport.zw);
}

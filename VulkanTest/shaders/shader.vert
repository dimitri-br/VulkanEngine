#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 lightPos;
    vec3 camPos;
} ubo;




layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 vNormal;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 Normal;
layout(location = 3) out vec3 FragPos;
layout(location = 4) out vec3 lightPos;
layout(location = 5) out vec3 viewPos;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // exports
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    Normal = mat3(transpose(inverse(ubo.model))) * vNormal;
    FragPos = vec3(ubo.model * vec4(inPosition, 1.0));
    lightPos = ubo.lightPos;
    viewPos = ubo.camPos;
}
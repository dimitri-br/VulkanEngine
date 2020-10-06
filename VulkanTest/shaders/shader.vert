#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 lightSpace;
    vec3 lightPos;
    vec3 camPos;
} ubo;

layout(binding = 2) uniform Material{
    float shininess;
    vec3 color;
} mat;

layout(binding = 3) uniform Transform{
    mat4 transform;

} trans;

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
layout(location = 6) out float shininess;
layout(location = 7) out vec3 outLightVec;
layout(location = 8) out vec4 outShadowCoord;
layout(location = 9) out vec3 outViewVec;







void main() {


    
    

    
    // exports
    fragColor = mat.color;
    fragTexCoord = inTexCoord;
    Normal = mat3(trans.transform) * vNormal;  

    FragPos = vec3(trans.transform * vec4(inPosition, 1.0));
    
    lightPos = ubo.lightPos;
    viewPos = normalize(-ubo.camPos);
    
    shininess = mat.shininess;
    
    vec4 pos = trans.transform * vec4(inPosition, 1.0);

    outViewVec = -pos.xyz;	

    outLightVec = normalize(ubo.lightPos - inPosition);
    //outLightVec = ubo.lightPos;
    //outShadowCoord = (ubo.lightSpace * trans.transform) * vec4(inPosition, 1.0);
    outShadowCoord = (ubo.lightSpace * trans.transform) * vec4(inPosition, 1.0);


    gl_Position = ubo.proj * ubo.view * trans.transform * vec4(inPosition, 1.0);;

}
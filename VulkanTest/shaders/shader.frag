#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 Normal;
layout(location = 3) in vec3 FragPos;
layout(location = 4) in vec3 lightPos;
layout(location = 5) in vec3 viewPos;
layout(location = 6) in float shininess;


layout(location = 0) out vec4 outColor;

void main() {
    // Define any constants here
    //vec3 lightPos = vec3(-4.0, 0.0, -5.0);
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    float specularStrength = shininess;

    // Normalize the normal ()
    vec3 norm = normalize(Normal);

    // Get the light direction ( to travel to)
    vec3 lightDir = normalize(lightPos - FragPos); 

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // Calculate specular
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  

    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor; 

    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
    vec3 result = (ambient + diffuse + specular) * fragColor;
    outColor = vec4(result * texture(texSampler, fragTexCoord).rgb, 1.0);
}
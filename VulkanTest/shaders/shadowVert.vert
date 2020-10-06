#version 450

layout (location = 0) in vec3 inPos;

layout(binding = 5) uniform UBO 
{
	mat4 depthMVP;
} ubo;
 
layout (binding = 3) uniform Transform
{
	mat4 transform;
} trans;


out gl_PerVertex 
{
    vec4 gl_Position;   
};

void main()
{    
	gl_Position =  (ubo.depthMVP * trans.transform) * vec4(inPos, 1.0);
}
#version 450

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 4) uniform sampler2D shadowMap;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 Normal;
layout(location = 3) in vec3 FragPos;
layout(location = 4) in vec3 lightPos;
layout(location = 5) in vec3 viewPos;
layout(location = 6) in float shininess;
layout (location = 7) in vec3 inLightVec;
layout (location = 8) in vec4 inShadowCoord;
layout (location = 9) in vec3 inViewVec;

layout(location = 0) out vec4 outColor;

#define ambient 0.003

const mat4 depthBias = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );
float CalcShadowFactor(vec4 lightspace_Position)
{

    vec4 ProjectionCoords = vec4(lightspace_Position.xyz / lightspace_Position.w, 1.0f);

    vec4 projCoords = depthBias * ProjectionCoords;



    if(texture(shadowMap, projCoords.xy).r < projCoords.z) return 0.5;
    else return 1.0;
}

float CalcShadow(){
    float closestDepth = shadow2DProj(shadowMap, ShadowCoords).r;
    float bias = 0.005;
    float currentDepth = ShadowCoords.z;
    float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;
    return shadow;
}

float textureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
		float dist = texture( shadowMap, shadowCoord.st + off ).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z ) 
		{
			shadow = ambient;
		}
	}
	return shadow;
}
float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowMap, 0);
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++)
	{
		for (int y = -range; y <= range; y++)
		{
			shadowFactor += textureProj(sc, vec2(dx*x, dy*y));
			count++;
		}
	
	}
	return shadowFactor / count;
}


void main(void)
{

    vec3 normal = normalize(Normal);

    vec3 light_Direction = normalize(lightPos - FragPos);

    vec3 half_vector = normalize(viewPos + light_Direction);

    float fndotl = dot(normal, light_Direction);
    float shadowFactor = CalcShadowFactor(inShadowCoord);

    float diffuse = max(0.0, fndotl) * shadowFactor + ambient;
    vec3 temp_Color = vec3(diffuse);

    float specular = max( 0.0, dot( normal, half_vector) );

    if(shadowFactor > 0.9){
        float fspecular = pow(specular, 64.0);
        temp_Color += fspecular + ambient;
    }

    outColor = vec4(shadowFactor * texture(texSampler, fragTexCoord).xyz * temp_Color, 1.0);
    //outColor = vec4(texture(texSampler, fragTexCoord).xyz * temp_Color, 1.0);
    
    	

}
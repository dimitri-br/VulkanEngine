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

#define ambient 0.01
#define depthbias false
const mat4 depthBias = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );


float textureProj(vec4 shadowCoord, vec2 off)
{
	float shadow = 1.0;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{

		
		float dist = texture(shadowMap, shadowCoord.st + off ).r;

		vec3 normal = normalize(Normal);
    	vec3 lightDir = normalize(lightPos - FragPos);
		//float bias = 0.005*tan(acos(dot(normal, lightDir))); // cosTheta is dot( n,l ), clamped between 0 and 1
		//bias = clamp(bias, 0,0.01);
    	//float bias = max(0.01 * (1.0 - dot(normal, lightDir)), 0.0005);
    	float bias = 0.0005;

		if ( shadowCoord.w > 0.0 && dist  < shadowCoord.z - bias) 
		{

			shadow = ambient;

		}
	}

	return shadow;
}
float filterPCF(vec4 sc)
{
	ivec2 texDim = textureSize(shadowMap, 0);
	//float scale = 1.5;
	float scale = 1.5;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 8;
	

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


float compute_shadow_factor(vec4 light_space_pos, sampler2D shadow_map, uint shadow_map_size, uint pcf_size)
{
   vec3 light_space_ndc = light_space_pos.xyz /= light_space_pos.w;
 
   if (abs(light_space_ndc.x) > 1.0 ||
       abs(light_space_ndc.y) > 1.0 ||
       abs(light_space_ndc.z) > 1.0)
      return 0.0;
 
   vec2 shadow_map_coord = light_space_ndc.xy * 0.5 + 0.5;
 
   // compute total number of samples to take from the shadow map
   int pcf_size_minus_1 = int(pcf_size - 1);
   float kernel_size = 2.0 * pcf_size_minus_1 + 1.0;
   float num_samples = kernel_size * kernel_size;
 
   // Counter for the shadow map samples not in the shadow
   float lighted_count = 0.0;
 
   // Take samples from the shadow map
   float shadow_map_texel_size = 1.0 / shadow_map_size;
   for (int x = -pcf_size_minus_1; x <= pcf_size_minus_1; x++)
   for (int y = -pcf_size_minus_1; y <= pcf_size_minus_1; y++) {
      // Compute coordinate for this PFC sample
      vec2 pcf_coord = shadow_map_coord + vec2(x, y) * shadow_map_texel_size;
 
      // Check if the sample is in light or in the shadow
      if (light_space_ndc.z <= texture(shadow_map, pcf_coord.xy).x)
         lighted_count += 1.0;
   }
 
   return lighted_count / num_samples;
}

void main(void)
{
	bool usePCF = true;

	vec3 color = texture(texSampler, fragTexCoord).rgb;
    vec3 normal = normalize(Normal);

    vec3 light_Direction = normalize(lightPos - FragPos);

    vec3 half_vector = normalize(viewPos + light_Direction);

    float fndotl = dot(normal, light_Direction);
	vec4 shadowCoord = inShadowCoord;
	if (depthbias){
		vec4 shadowCoord = inShadowCoord * depthBias;
	}
    //float shadowFactor = (usePCF) ? filterPCF(shadowCoord/shadowCoord.w) : textureProj(inShadowCoord/inShadowCoord.w, vec2(0.0f));
    float shadowFactor = compute_shadow_factor(shadowCoord, shadowMap, 4096, 8);

    float diffuse = max(0.0, fndotl) + ambient;
    vec3 temp_Color = vec3(diffuse);

    float specular = max( 0.0, dot( normal, half_vector) );
    

    if(shadowFactor > 0.9){
        float fspecular = pow(specular, 64.0);
        temp_Color += fspecular + ambient;
    }
	
    outColor = vec4(shadowFactor * texture(texSampler, fragTexCoord).xyz * temp_Color, 1.0);
    
    	

}
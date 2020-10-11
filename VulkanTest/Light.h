#pragma once
#include "lightTypes.h"
#include <glm/glm.hpp> // Needed for math functions like matrices etc
#include <glm/gtc/matrix_transform.hpp> // transformations
#include <glm/gtx/rotate_vector.hpp>

class Light
{
public:
	lightType type;
	lightUpdate update;
	glm::vec3 position;
	glm::vec3 rotation;

	glm::mat4 lightView;

	float lightFOV = 45.0f;


	// Keep depth range as small as possible
	// for better shadow map precision
	float zNear = 2.0f;
	float zFar = 96.0f;

	// So we can selectively rerender shadowmaps
	bool hasRendered = false;

	// Shadowmap stuff
	VkImage shadowImage;
	VkDeviceMemory shadowImageMemory;
	VkImageView shadowImageView;
	VkSampler shadowSampler;

	void init(lightType lighttype, lightUpdate updateRate, glm::vec3 pos, glm::vec3 rot);

	void calculateView();
};


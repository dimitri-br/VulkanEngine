#pragma once
#include "lightTypes.h"
#include <glm/glm.hpp> // Needed for math functions like matrices etc
#include <glm/gtc/matrix_transform.hpp> // transformations
#include <glm/gtx/rotate_vector.hpp>

class Light
{
public:
	lightType type;
	glm::vec3 position;
	glm::vec3 rotation;

	void init(lightType lighttype, glm::vec3 pos, glm::vec3 rot);


};


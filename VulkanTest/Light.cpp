#include "Light.h"


void Light::init(lightType lighttype, glm::vec3 pos, glm::vec3 rot) {
	type = lighttype;
	position = pos;
	rotation = rot;
}
#define GLFW_INCLUDE_VULKAN 
#include <GLFW/glfw3.h> // Windowing
#include <iostream> // IDK
#include <stdexcept> // Errors
#include <cstdlib> // STD lib
#include <vector> // For arrays
#include <glm/glm.hpp> // Needed for math functions like matrices etc
#include <glm/gtc/matrix_transform.hpp> // transformations
#include <glm/gtx/rotate_vector.hpp>
#include "object.h" // holds local files
#include "transform.h"
#include "material.h"
#include "Vertex.h"



void Object::init(std::string model, std::string texture, Material material, Transform trans)
{
	model_path = model;
	texture_path = texture;
	mat = material;
	transform = trans;
}

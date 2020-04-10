// ----------------------------------------------------------------------------------------------------
// 
// 	File name: AreaLight.h
//	Created By: Haard Panchal
//	Create Date: 03/16/2020
//
//	Description:
//		File has the definition and implementation of the AreaLight class
//      The point light disperses ray from a single point p_0
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef AREALIGHTH
#define AREALIGHTH

#include <curand_kernel.h>

#include "Vector3.h"
#include "Light.h"

class AreaLight : public Light {
    private:
        Vector3 c_0; // Position of the origin of the area
        Vector3 c_c; // Position of the center of the area
        Vector3 n_0, n_1; // Normal to the area
        Vector3 a_n, a_up; // The up vector for the area
        float w, h;   // Width and height of the area 

        Vector3 current_position; // The position of the current sample of the area light
        curandState * rand_state; // CUDA random state

    public:
        __host__ __device__ AreaLight();
        __host__ __device__ AreaLight(Vector3& position, Vector3& normal, Vector3& up, float width, float height, curandState& random_state);
        __host__ __device__ ~AreaLight();

        __device__ void setRandomSamplePosition();
        __device__ Vector3 getLightAtPoint(Vector3& point);
        __host__ __device__ Vector3 getLightPosition() { return current_position; }
};

__host__ __device__ AreaLight::AreaLight() {}

__host__ __device__ AreaLight::AreaLight(Vector3& position, Vector3& normal, Vector3& up, float width, float height, curandState& random_state) 
 : c_c(position), a_n(normal), a_up(up), w(width), h(height), rand_state(&random_state) {
    a_n.make_unit_vector();
    a_up.make_unit_vector();
    
    n_0 = cross(a_n, a_up);
    n_0.make_unit_vector();

    n_1 = cross(n_0, a_n);
    n_1.make_unit_vector();

    c_0 = c_c - (w / 2.0f) * n_0 - (h / 2.0f) * n_1;
    current_position = c_c;
}

__host__ __device__ AreaLight::~AreaLight() {}

__device__ Vector3 AreaLight::getLightAtPoint(Vector3& point) {
    setRandomSamplePosition();
    Vector3 p = point - current_position;
    p.make_unit_vector();
    return p; 
}

__device__ void AreaLight::setRandomSamplePosition() {
    float x = curand_uniform(rand_state) * w;
    float y = curand_uniform(rand_state) * h;

    current_position = c_0 + x * n_0 + y * n_1;
}

#endif
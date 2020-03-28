// ----------------------------------------------------------------------------------------------------
// 
// 	File name: DirectionalLight.h
//	Created By: Haard Panchal
//	Create Date: 03/16/2020
//
//	Description:
//		File has the definition and implementation of the DirectionalLight class
//      A directional light means that the light is coming with 
//      equal intensity and direction at all the points.
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
// ----------------------------------------------------------------------------------------------------

#ifndef DIRECTIONALLIGHTH
#define DIRECTIONALLIGHTH

#include "Vector3.h"
#include "Light.h"

class DirectionalLight: public Light {
    private:
        Vector3 n; // Direction of the light

    public:
        __host__ __device__ DirectionalLight();
        __host__ __device__ DirectionalLight(Vector3& direction);
        __host__ __device__ ~DirectionalLight();

        __host__ __device__ Vector3 getLightAtPoint(Vector3& point) const;
};

__host__ __device__ DirectionalLight::DirectionalLight() {}

__host__ __device__ DirectionalLight::DirectionalLight(Vector3& direction) {
    n = direction;
    n.make_unit_vector();
}

__host__ __device__ DirectionalLight::~DirectionalLight() {}

__host__ __device__ Vector3 DirectionalLight::getLightAtPoint(Vector3& point) const {
    return n; 
}

#endif
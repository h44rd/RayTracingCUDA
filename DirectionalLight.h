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
#include "Helpers.h"

class DirectionalLight: public Light {
    private:
        Vector3 n; // Direction of the light

    public:
        __device__ DirectionalLight();
        __device__ DirectionalLight(Vector3& direction);
        __device__ ~DirectionalLight();

        __device__ Vector3 getLightAtPoint(Vector3& point);
        __device__ Vector3 getLightPosition() { return LARGENUMBER * n; } // The direction is multiplied with a large number to indicate 
                                                                                   // light infinitly away
};

__device__ DirectionalLight::DirectionalLight() {}

__device__ DirectionalLight::DirectionalLight(Vector3& direction) {
    n = direction;
    n.make_unit_vector();
}

__device__ DirectionalLight::~DirectionalLight() {}

__device__ Vector3 DirectionalLight::getLightAtPoint(Vector3& point) {
    return n; 
}

#endif
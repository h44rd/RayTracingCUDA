// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Light.h
//	Created By: Haard Panchal
//	Create Date: 03/16/2020
//
//	Description:
//		File has the definition and implementation of the abstract Light class
//      This is the base class for all the types of lights.
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef LIGHTH
#define LIGHTH

#include "Vector3.h"

class Light {
    private:

    public:
        __device__ Light();
        __device__ ~Light();

        // This function has to be overridden for each type of light
        // The returned vector v: |v| <= 1
        __device__ virtual Vector3 getLightAtPoint(Vector3& point) = 0;
        
        __device__ virtual Vector3 getLightPosition();
        __device__ virtual bool ifSamplingRequired();
        __device__ virtual void setRandomSamplePosition(curandState& rand_state);
};

__device__ Light::Light() {}

__device__ Light::~Light() {}

__device__ Vector3 Light::getLightPosition() {
    return Vector3(0.0f, 0.0f, 0.0f);
}

__device__ bool Light::ifSamplingRequired() {
    return false;
}

__device__ void Light::setRandomSamplePosition(curandState& rand_state) {
    // empty
}

#endif

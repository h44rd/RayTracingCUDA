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
        __host__ Light();
        __host__ ~Light();

        // This function has to be overridden for each type of light
        // The returned vector v: |v| <= 1
        __host__ virtual Vector3 getLightAtPoint(Vector3& point) const = 0;
};

__host__ Light::Light() {}

__host__ Light::~Light() {}

#endif
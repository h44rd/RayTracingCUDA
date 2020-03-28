// ----------------------------------------------------------------------------------------------------
// 
// 	File name: PointLight.h
//	Created By: Haard Panchal
//	Create Date: 03/16/2020
//
//	Description:
//		File has the definition and implementation of the PointLight class
//      The point light disperses ray from a single point p_0
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef POINTLIGHTH
#define POINTLIGHTH

#include "Vector3.h"
#include "Light.h"

class PointLight: public Light {
    private:
        Vector3 p_0; // Position of the light

    public:
        __host__ __device__ PointLight();
        __host__ __device__ PointLight(Vector3& position);
        __host__ __device__ ~PointLight();

        __host__ __device__ Vector3 getLightAtPoint(Vector3& point) const;
};

__host__ __device__ PointLight::PointLight() {}

__host__ __device__ PointLight::PointLight(Vector3& position) : p_0(position) {}

__host__ __device__ PointLight::~PointLight() {}

__host__ __device__ Vector3 PointLight::getLightAtPoint(Vector3& point) const {
    Vector3 p = point - p_0;
    p.make_unit_vector();
    return p; 
}

#endif
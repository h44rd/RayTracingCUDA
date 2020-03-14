// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Ray.h
//	Created By: Haard Panchal
//	Create Date: 03/11/2020
//
//	Description:
//		File has the definition and implementation of the Ray class
//      Ray is define as:
//      P = P_0 + t * N;
//      where P is the arbitrary point, P_0 is the initial point and N is direction of the ray
//      t is the parameter
//
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef RAYH
#define RAYH

#include "Vector3.h"

/*  Class: Ray
//
//	Defines the implementation of the Ray.
//
//	Constructor Parameters:
//   TODO
//	
//	Return:
//   TODO
*/

class Ray {
    private:
        Vector3 p_0; // The starting point of the ray
        Vector3 n; // The direction of the ray

    public:
        __host__ __device__ Ray();
        __host__ __device__ Ray(const Vector3& point, const Vector3& direction);
        __host__ __device__ ~Ray();
        
        __host__ __device__ Vector3 getPoint(float t); // Get a point t distance away from the starting point
        __host__ __device__ Vector3 getStartingPoint(); // Get the starting point of the ray
        __host__ __device__ Vector3 getDirection(); // Get the direction of the ray
};

__host__ __device__ Ray::Ray() {}

__host__ __device__ Ray::Ray(const Vector3& point, const Vector3& direction) : p_0(point) {
    n = direction;
    n.make_unit_vector();
}

__host__ __device__ inline Vector3 Ray::getPoint(float t) {
    return p_0 + t * n; // Returns the point calculated using the passed paramter
}


#endif
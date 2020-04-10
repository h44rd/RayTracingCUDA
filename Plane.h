// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Plane.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the Plane class
//      Plane is defined by:
//          n_0: normal
//          p_i: Given point on the plane
//      Therefore, (p - p_i).n_0 = 0 for any arbitrary point p on the plane
//      
//
//	History:
//		03/14/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef PLANEH
#define PLANEH

#include "Vector3.h"
#include "VisibleObject.h"


class Plane : public VisibleObject {
    private:
        Vector3 n_0; // Normal to the plane
        Vector3 p_i; // A given point on the plane
        Vector3 c_0; // Color of the plane {{{{COMING SOON: ***MATERIALS***}}}}

    public:
        __host__ __device__ Plane();
        __host__ __device__ Plane(Vector3& normal, Vector3& point, Vector3& color);
        __host__ __device__ ~Plane();

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : if hit (+ve if hit, -ve otherwise)
        __host__ __device__ Vector3 getIntersectInfo(const Ray& incoming) const;

        // The normal to the plane
        __host__ __device__ Vector3 getNormalAtPoint(Vector3& point) const { return n_0; }

        __host__ __device__ Vector3 getColor(Vector3& point) const { return c_0; }
};

__host__ __device__ Plane::Plane() {}

__host__ __device__ Plane::Plane(Vector3& normal, Vector3& point, Vector3& color) : p_i(point), c_0(color) {
    n_0 = normal;
    n_0.make_unit_vector();
}

__host__ __device__ Plane::~Plane() {}

__host__ __device__ Vector3 Plane::getIntersectInfo(const Ray& incoming) const {
    Vector3 intersection(0.0f, 0.0f, 0.0f);

    float slope = dot(incoming.getDirection(), n_0);
    float t = 0.0f;
    float ifIntersect = 0.0f;

    bool ifAngled = false;

    if(abs(slope) > 0.000001f) {
        t = dot(p_i - incoming.getStartingPoint(), n_0) / slope;
        ifAngled = true;
    } 

    #ifdef DEBUG
    std::cout<<"Plane t: "<<t<<std::endl;
    std::cout<<"Slope: "<<slope<<std::endl;
    #endif
    
    if(t >= 0.0f && ifAngled) {
        ifIntersect = 1.0f;
    } else {
        ifIntersect = -1.0f;
    }

    intersection[0] = t;
    intersection[2] = ifIntersect;

    return intersection;
}

#endif
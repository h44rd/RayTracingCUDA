// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Sphere.h
//	Created By: Haard Panchal
//	Create Date: 03/14/2020
//
//	Description:
//		File has the definition and implementation of the Sphere class
//      Sphere is define as:
//      (P - P_c).(P - P_c) - r*r <= 0
//      where P is the arbitrary point, P_c is the center of the sphere and r is the radius of the sphere
//
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef SPHEREH
#define SPHEREH

#include<math.h>

#include "Helpers.h"
#include "VisibleObject.h"
#include "Vector3.h"

class Sphere : public VisibleObject {
    private:
        float r; // Radius
        Vector3 p_c; // Center 
        Vector3 c_0; // Color

    public:
        __host__ __device__ Sphere();
        __host__ __device__ Sphere(Vector3& center, float radius, Vector3& color);
        __host__ __device__ ~Sphere();

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : if hit (+ve if hit, -ve otherwise)
        __host__ __device__ Vector3 getIntersectInfo(const Ray& incoming) const;

        // The normal to the sphere 
        __host__ __device__ Vector3 getNormalAtPoint(Vector3& point) const { return (point - p_c)/r; }

        // The color
        __device__ Vector3 getColor(Vector3& point) const;

        __device__ __host__ int getTypeID() { return SPHERE_TYPE_ID; }

        __device__ void update();
};

__host__ __device__ Sphere::Sphere() {}

__host__ __device__ Sphere::Sphere(Vector3& center, float radius, Vector3& color) : p_c(center), r(radius), c_0(color) {}

__host__ __device__ Sphere::~Sphere() {}

/*  Function: getIntersectInfo for sphere
//
//  Solves the Equation:
//	[1] * t * t + 2 * [n . (p_0 - p_c)] * t + [(p_0 - p_c).(p_0 - p_c) - r * r]= 0
//   a                     b                                  c
// 
//  t = -b +/- sqrt(b * b  - c)
//	
//	Return:
//   Vector3 v;
//   v.x = solution t
//   v.y = dot product (intensity) of ray with normal at the point
//   v.z = if intersection happened v.z > 0
*/
__host__ __device__ Vector3 Sphere::getIntersectInfo(const Ray& incoming) const {
    Vector3 intersection(0.0f, 0.0f, 0.0f);

    #ifdef DEBUG
    std::cout<<"Ray direction: "<<incoming.getDirection()<<std::endl;
    std::cout<<"Ray starting point: "<<incoming.getStartingPoint()<<std::endl;
    std::cout<<"Sphere center: "<<incoming.getStartingPoint() - p_c<<std::endl;
    #endif

    float a = 1.0f;
    
    float b = dot(incoming.getDirection(), incoming.getStartingPoint() - p_c);
    
    float c = dot(incoming.getStartingPoint() - p_c, incoming.getStartingPoint() - p_c) - r * r;

    float discriminant = b * b - c;
    float t = 0.0f;
    float slope = 0.0f;
    float ifIntersect = 0.0f;
    float ifInside = -1.0f;

    // Checking if the ray intersects AND b <= 0 makes sure that the ray is not pointing away from the center of the sphere
    if(discriminant > 0.0f && b <= 0.0f) {
        t = -1.0f * b - sqrt(discriminant);

        if(t <= 0.0f) {
            t = -1.0f * b + sqrt(discriminant);
            ifInside = 1.0f;
        }

        ifIntersect = 1.0;
    } else {
        t = -1.0;
        ifIntersect = -1.0;
    }
    #ifdef DEBUG
    std::cout<<"c: "<<c<<std::endl;
    std::cout<<"b: "<<b<<std::endl;
    std::cout<<"Discriminant: "<<discriminant<<std::endl;
    std::cout<<"Sphere t: "<<t<<std::endl;
    #endif
    intersection[0] = t;
    intersection[1] = ifInside;
    intersection[2] = ifIntersect;

    return intersection;  
}

__device__ Vector3 Sphere::getColor(Vector3& point) const {
    if(m != NULL) {
        float theta = atan2(-1 * (point.z() - p_c.z()) , point.x() - p_c.x());
        float u = (theta + PI) / (2.0 * PI);
        float phi = acos(-1 * (point.y() - p_c.y()) / r);
        float v = phi / PI;
        #ifdef MATERIALDEBUG
            printf("py: %f. y: %f, distance: %f, phi: %f\n", point.y(), c_0.y(), (point - p_c).length(), phi);
        #endif
        return m->getBilinearColor(u, v);
    }
    return c_0;
}

__device__ void Sphere::update() {
    p_c += Vector3(0.0f, 0.1f, 0.0f);
}
#endif
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

#include "VisibleObject.h"
#include "Vector3.h"

class Sphere : public VisibleObject {
    private:
        float r;
        Vector3 p_c;
        Vector3 c_0;
    
    public:
        __host__ Sphere();
        __host__ Sphere(Vector3& center, float radius, Vector3& color);
        __host__ ~Sphere();

        // The function will return a Vector3 with x : Parameter t, y : slope of hit, z : if hit (+ve if hit, -ve otherwise)
        __host__ Vector3 getIntersectInfo(const Ray& incoming) const;

        // The normal to the sphere 
        __host__ Vector3 getNormalAtPoint(Vector3& point) const { return (point - p_c)/r; }

        // The color
        __host__ Vector3 getColor(Vector3& point) const { return c_0; }
};

__host__ Sphere::Sphere() {}

__host__ Sphere::Sphere(Vector3& center, float radius, Vector3& color) : p_c(center), r(radius), c_0(color) {}

__host__ Sphere::~Sphere() {}

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
//   v.z = if intersection happened
*/
__host__ Vector3 Sphere::getIntersectInfo(const Ray& incoming) const {
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

    // Checking if the ray intersects AND b <= 0 makes sure that the ray is not pointing away from the center of the sphere
    if(discriminant > 0.0f && b <= 0.0f) {
        t = -1.0f * b - sqrt(discriminant);
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
    intersection[2] = ifIntersect;

    return intersection;  
}

#endif
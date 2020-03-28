// ----------------------------------------------------------------------------------------------------
// 
// 	File name: Camera.h
//	Created By: Haard Panchal
//	Create Date: 03/11/2020
//
//	Description:
//		File has the definition and implementation of the Camera class
//
//	History:
//		03/13/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#ifndef CAMERAH
#define CAMERAH

#include "Vector3.h"
#include "Ray.h"

/*  Class: Camera
//
//	Defines the implementation of the Camera, the eye to our virtual world.
//
//	Constructor Parameters:
//   TODO
//	
//	Return:
//   TODO
*/

class Camera
{
    private:
        Vector3 p_e; // Position of the camera
        Vector3 p_c; // Position of the center of the screen
        Vector3 p_00; // Bottom Left corner of the screen 
        Vector3 v_view, v_up; // View and Up vectors that defines the orientation of the camera
        Vector3 n_0, n_1; // unit Perpendicular vectors that define the screen
        Vector3 n_2; // n2 : v_view but unit vector

        float s_x, s_y; // Dimensions of the screen 
        float d; // Distance of the screen from the camera
        
    public:
        __host__ __device__ Camera();
        __host__ __device__ Camera(const Vector3& position, const Vector3& direction, const Vector3& up, float sx, float sy, float screen_distance);
        __host__ __device__ ~Camera();

        __host__ __device__ Ray getRay(float u, float v) const; // Get the ray corresponding the u,v cordinates on the screen
        __host__ __device__ inline Vector3 getUnitViewVector() const { return n_2; }
};

__host__ __device__ Camera::Camera() {}

__host__ __device__ Camera::Camera(const Vector3& position, const Vector3& direction, const Vector3& up, float sx, float sy, float screen_distance) : p_e(position), v_view(direction), v_up(up), s_x(sx), s_y(sy), d(screen_distance) {
    n_0 = cross(v_view, v_up);
    n_0.make_unit_vector();

    n_1 = cross(n_0, v_view);
    n_1.make_unit_vector();

    n_2 = unit_vector(v_view);

    p_c = p_e + n_2 * d; // Going to the center of the screen from the camera position in the direction of the view vector

    p_00 = p_c - (s_x / 2.0f) * n_0 - (s_y  / 2.0f) * n_1; // The 0,0 for the screen : Bottom Left corner

    #ifdef INITDEBUG
    std::cout<<"n2: "<<n_2<<std::endl;
    std::cout<<"p00: "<<p_00<<std::endl;
    std::cout<<"pe: "<<p_e<<std::endl;
    std::cout<<"pc: "<<p_c<<std::endl;
    #endif
}

Camera::~Camera() {}

/*  Function: getRay
//
//	Get the ray at a point u,v on the screen
//  Ray starts from p_e and direction is from p_e to the (p_00 + (u * n_0) + (v * n_1))
//
//	Function Parameters:
//   TODO
//	
//	Return:
//   TODO
*/
__host__ __device__ inline Ray Camera::getRay(float u, float v) const {
    #ifdef DEBUG
    std::cout<<"Ray direction in getRay: "<<(p_00 + (u * n_0 * s_x) + (v * n_1 * s_y)) - p_e<<std::endl;
    Vector3 r = (p_00 + (u * n_0 * s_x) + (v * n_1 * s_y)) - p_e;
    r.make_unit_vector();
    std::cout<<"Ray direction in getRay unit vector: "<<r<<std::endl;
    std::cout<<"n2: "<<n_2<<std::endl;
    std::cout<<"p00: "<<p_00<<std::endl;
    std::cout<<"pe: "<<p_e<<std::endl;
    #endif
    return Ray(p_e, (p_00 + (u * n_0 * s_x) + (v * n_1 * s_y)) - p_e);
}

#endif
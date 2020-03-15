// ----------------------------------------------------------------------------------------------------
// 
// 	File name: main.cu
//	Created By: Haard Panchal
//	Create Date: 03/11/2020
//
//	Description:
//		Main file for the Ray Tracing project 
//
//	History:
//		03/10/19: H. Panchal Created the file
//
//  Declaration:
//      N/A
//
// ----------------------------------------------------------------------------------------------------

#include <iostream>
#include <math.h>

#include "Vector3.h"
#include "Ray.h"

#include "Camera.h"
#include "World.h"

#include "Sphere.h"
#include "Plane.h"

/*  Function: main
//
//	Parses the argument list. Initializes the relevant objects and starts rendering.
//
//	Parameters:
//
//		int argc: Number of arguments
//		char *argv[]: List of the arguments
//	
//	Return:
//		int: 0 if successful
*/

int main(int argc, char *argv[]) {
    
    Vector3 a(1, 2, 3);
    Vector3 b(5, 6, 7);

    Vector3 c = b;

    std::cout<<c<<std::endl;
    std::cout<<dot(a,c)<<std::endl;

    Vector3 color(1, 1, 1);

    Vector3 center(0.0, 1.0, 2.0);
    float r = 10.0f;
    Sphere s(center, r, color);

    Vector3 point(2.0, 3.0, 5.0);
    Vector3 normal(-1, -1, -1);
    Plane p(point, normal, color);

    World w;
    w.addVisibleObject(&s);
    w.addVisibleObject(&p);

    return 0;
}
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

// #define RENDERDEBUG
#define ACTUALRENDER
// #define INITDEBUG

#include <iostream>
#include <math.h>

#include "Vector3.h"
#include "Ray.h"

#include "Camera.h"
#include "World.h"

#include "Sphere.h"
#include "Plane.h"

#include "RenderEngine.h"
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
    
    Vector3 color1(1.0f, 0.5f, 1.0f);
    Vector3 color2(0.5f, 1.0f, 0.25f);
    Vector3 color3(0.2f, 0.3f, 0.7f);
    Vector3 color4(0.9f, 0.0f, 0.1f);

    Vector3 center(-1.0, 0.0, 0.0);
    float r = 1.0f;
    Sphere s(center, r, color1);

    Vector3 center2(1.0, 0.0, 0.0);
    float r2 = 1.0f;
    Sphere s2(center2, r2, color4);

    Vector3 point(0.0, -1.0, 0.0);
    Vector3 normal(0, 0.7, 0.1);
    Plane p(normal, point, color2);

    Vector3 point2(1.0, 0.0, 0.0);
    Vector3 normal2(-0.7, -0.3, 0.1);
    Plane p2(normal2, point2, color3);

    Vector3 positioncam(0.0, 0.0, 5.0);
    Vector3 direction = - positioncam;
    Vector3 updir(0.0, 1.0, 0.0);
    Camera cam(positioncam, direction, updir, 1.0, 1.0, 1.0);

    Vector3 light(0.5f,0.7f,3.0f); // Position of the light
    // light.make_unit_vector();

    World w;
    w.addVisibleObject(&s);
    w.addVisibleObject(&s2);
    w.addVisibleObject(&p);
    w.addVisibleObject(&p2);
    w.setCamera(cam);
    w.setLight(light);

    int wid = 1200, hgt = 1200;
    RenderEngine r_engine(wid, hgt, w);

    r_engine.renderAllPixels();

    return 0;
}
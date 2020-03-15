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

    Vector3 center(0.0, 1.0, 2.0);
    float r = 10.0f;
    Sphere s(center, r, color1);

    Vector3 point(2.0, 3.0, 5.0);
    Vector3 normal(-1, -1, -1);
    Plane p(point, normal, color2);

    Vector3 positioncam(0.0, 0.0, -5.0);
    Vector3 direction = center - positioncam;
    Vector3 updir(0.0, 1.0, 0.0);
    Camera cam(positioncam, direction, updir, 1.0, 1.0, 1.0);

    Vector3 light(1.0f,1.0f,1.0f);
    light.make_unit_vector();

    World w;
    w.addVisibleObject(&s);
    w.addVisibleObject(&p);
    w.setCamera(cam);
    w.setLight(light);

    int wid = 1200, hgt = 1200;
    RenderEngine r_engine(wid, hgt, w);

    r_engine.renderAllPixels();

    return 0;
}
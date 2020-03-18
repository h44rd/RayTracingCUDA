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
#include "PointLight.h"
#include "DirectionalLight.h"
#include "SpotLight.h"

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

    // const float PI = 3.1415927;
    
    Vector3 color1(1.0f, 0.5f, 1.0f);
    Vector3 color2(0.5f, 1.0f, 0.25f);
    Vector3 color3(0.2f, 0.3f, 0.7f);
    Vector3 color4(0.9f, 0.0f, 0.1f);
    Vector3 color5(0.5f, 0.4f, 0.2f);

    Vector3 center(-1.0, 0.0, 0.0);
    float r = 1.0f;
    Sphere s(center, r, color1);

    Vector3 center2(0.5, 0.5, 0.0);
    float r2 = .25f;
    Sphere s2(center2, r2, color4);

    Vector3 center3(0.5, -0.5, 0.0);
    float r3 = .5f;
    Sphere s3(center3, r3, color4);

    Vector3 point(0.0, -2.5, 0.0);
    Vector3 normal(0, 1.0, 0.0);
    Plane p(normal, point, color2);

    Vector3 point2(0.0, 2.5, 0.0);
    Vector3 normal2(0.0, -1.0, 0.0);
    Plane p2(normal2, point2, color3);

    Vector3 point3(0.0, 1.0, 0.0);
    Vector3 normal3(-0.0, -0.7, 0.1);
    Plane p3(normal3, point3, color5);

    Vector3 positioncam(0.0, 0.0, 5.0);
    Vector3 lookat(0.0f, 0.0f, 0.0f);
    Vector3 direction = lookat - positioncam;
    Vector3 updir(0.0, 1.0, 0.0);
    Camera cam(positioncam, direction, updir, 1.0, 1.0, 1.0);

    Vector3 lightpos(0.0f, .5f, 3.0f); // Position of the light
    PointLight pointlight(lightpos);

    Vector3 lightdir(-1.0f, -1.0f, -1.0f);
    DirectionalLight dirLight(lightdir);

    float beam_angle = 10.0;
    float falloff_angle = 30.0;
    beam_angle = beam_angle * PI / 180.0;
    falloff_angle = falloff_angle * PI / 180.0;
    Vector3 spotlightpos(-0.3, 0.25, 5.0f);
    Vector3 spotlightdir = -spotlightpos;
    SpotLight spotlight(spotlightpos, spotlightdir, beam_angle, falloff_angle);

    World w;
    // w.addVisibleObject(&s);
    // w.addVisibleObject(&s2);
    // w.addVisibleObject(&s3);
    w.addVisibleObject(&p);
    w.addVisibleObject(&p2);
    // w.addVisibleObject(&p3);
    w.setCamera(cam);
    w.addLight(&pointlight); // Light id 0
    w.addLight(&dirLight); // Light id 1
    w.addLight(&spotlight); // Light id 2
    w.setLightId(2);

    float segment = 20.0;
    float radius_circle = 2.0f;
    for(int i = 0; i < 360; i += segment) {
        float angle = i * PI / 180.0;
        Vector3* color_each = new Vector3(abs(sin(angle)), abs(cos(angle)), abs(sin(angle) * cos(angle)));
        Vector3* pos = new Vector3(radius_circle* sin(angle), radius_circle * cos(angle), 0.0);
        Sphere * sphere = new Sphere(*pos, 0.3, *color_each);
        w.addVisibleObject(sphere);
    }

    int wid = 1200, hgt = 1200;
    RenderEngine r_engine(wid, hgt, w);

    // r_engine.setSharpEdge(0.4, 0.6);
    r_engine.renderAllPixels();

    return 0;
}
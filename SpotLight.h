// ----------------------------------------------------------------------------------------------------
// 
// 	File name: SpotLight.h
//	Created By: Haard Panchal
//	Create Date: 03/16/2020
//
//	Description:
//		File has the definition and implementation of the SpotLight class
//      The spot light has light coming in the shape of a cone.
//      The light has a position p_0, a direction n, and an angle falloff_angle.
//      [ANGLE SHOULD BE GIVEN IN RADIANS]
//      
//	History:
//		03/16/19: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef SPOTLIGHTH
#define SPOTLIGHTH

#include "Vector3.h"
#include "Light.h"
#include "math.h"
#include "Helpers.h"

class SpotLight: public Light {
    private:

        Vector3 p_0; // Position of the light
        Vector3 n; // Direction of the light
        float falloff_angle; // The maximum angle the ray should make for it to exist [ANGLE SHOULD BE IN RADIANS]
        float beam_angle;

    public:
        __device__ SpotLight();
        __device__ SpotLight(Vector3& position, Vector3& direction, float p_beam_angle, float p_falloff_angle);
        __device__ ~SpotLight();

        __device__ Vector3 getLightAtPoint(Vector3& point);
        __device__ Vector3 getLightPosition() { return p_0; }
};

__device__ SpotLight::SpotLight() {}

__device__ SpotLight::SpotLight(Vector3& position, Vector3& direction, float p_beam_angle, float p_falloff_angle) : p_0(position), n(direction), beam_angle(p_beam_angle), falloff_angle(p_falloff_angle) {
    n.make_unit_vector();
}

__device__ SpotLight::~SpotLight() {}

__device__ Vector3 SpotLight::getLightAtPoint(Vector3& point) {
    Vector3 light_direction = point - p_0;
    float length_from_light = light_direction.length();
    light_direction.make_unit_vector();

    float angle = acos(dot(n, light_direction));
    
    // // The farther the light 

    length_from_light = smoothstep(0.0, 1.0, sqrt(length_from_light));
    // float step = smoothstep(0.0, length_from_light, (falloff_angle - angle)/falloff_angle);
    // return step * light_direction;

    float corrected_falloff = falloff_angle;

    if(angle < beam_angle) {
        return light_direction;
    } else if(angle < beam_angle + corrected_falloff) {
        angle = angle - beam_angle;
        float portion = (corrected_falloff - angle)/(corrected_falloff);
        return portion * light_direction;
    }
    return Vector3(0, 0, 0);

}

#endif
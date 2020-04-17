// ----------------------------------------------------------------------------------------------------
// 
// 	File name: TextureMaterial.h
//	Created By: Haard Panchal
//	Create Date: 04/11/2020
//
//	Description:
//		File has the definition and implementation of the TextureMaterial class
//      This class implements textures used for applying 2D image texture on objects.
//      
//      The texture at a UV point is given out.
//	History:
//		04/11/2020: H. Panchal Created the file
//
//  Declaration:
//
// ----------------------------------------------------------------------------------------------------

#ifndef TEXTUREMATERIALH
#define TEXTUREMATERIALH

#include "Vector3.h"
#include "Material.h"

class TextureMaterial : public Material {
    private:
        unsigned char * color_image;
        int c_i_height, c_i_width, c_i_channels;
    public:
        __device__ TextureMaterial();
        __device__ ~TextureMaterial();

        __device__ void setColorImage(int width, int height, int channels, unsigned char * image);
        
        __device__ Vector3 getColorAtIndex(int x, int y);
        __device__ Vector3 getBilinearColor(float u, float v); // Returns the color bilinealy interpolated color. IMP: 0 <= u,v <= 1, repeats otherwise
};

__device__ TextureMaterial::TextureMaterial() {}

__device__ TextureMaterial::~TextureMaterial() {}

__device__ void TextureMaterial::setColorImage(int width, int height, int channels, unsigned char * image) {
    c_i_width = width;
    c_i_height = height;
    c_i_channels = channels;
    
    color_image = image;
}

__device__ Vector3 TextureMaterial::getColorAtIndex(int x, int y) {
    if(x < c_i_width && y < c_i_height) {
        int index = c_i_channels * (c_i_width * y + x);
        
        // #ifdef MATERIALDEBUG
        //     printf("width: %d. height: %d, x: %d, y: %d\n", c_i_width, c_i_height, x, y);
        //     printf("r: %f. g: %f, b: %f\n", float(color_image[index + 0])/255.99f, float(color_image[index + 1])/255.99f, float(color_image[index + 2])/255.99f);
        // #endif

        return Vector3(float(color_image[index + 0])/255.99f, float(color_image[index + 1])/255.99f, float(color_image[index + 2])/255.99f);
    }
    return Vector3(0.0, 0.0, 0.0);
}

__device__ Vector3 TextureMaterial::getBilinearColor(float u, float v) {
    float scaled_u = fabs(u * c_i_width);
    float scaled_v = fabs(v * c_i_height);

    float tx = scaled_u - floor(scaled_u);
    float ty = scaled_v - floor(scaled_v);

    int x_0 = int(floor(scaled_u + 0.5));
    int y_0 = int(floor(scaled_v + 0.5));
    
    // #ifdef MATERIALDEBUG
    //         printf("x_0: %d. y_0: %d, scaled_u: %f, scaled_v: %f\n", x_0, y_0, u, v);
    // #endif
    
    return ty * (tx * getColorAtIndex(x_0 % c_i_width, y_0 % c_i_height) + (1.0 - tx) * getColorAtIndex((x_0 + 1) % c_i_width, y_0 % c_i_height))
           + (1.0 - ty) * (tx * getColorAtIndex(x_0 % c_i_width, (y_0 + 1) % c_i_height) + (1.0 - tx) * getColorAtIndex((x_0 + 1) % c_i_width, (y_0 + 1) % c_i_height));
}
#endif
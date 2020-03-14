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
#include "Camera.h"

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

    return 0;
}
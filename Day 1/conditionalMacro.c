#include <stdio.h>
#define PI 3.14159 

#define AREA(r)  (PI * (r * r)) // macro to calculate area of square

//#define r radius
// ifndef 
#ifndef radius // if AREA is not defined
#define radius 7
#endif 

// if elif
// we can only use integer constants in #if and #elif 

#if radius > 10// if AREA is not defined
#define radius 10
#elif radius < 5 
#define radius 20
#else
#define radius 7
#endif


int main(){


printf("Area of circle: %f\n", AREA(radius)); // prints 314.159000


}
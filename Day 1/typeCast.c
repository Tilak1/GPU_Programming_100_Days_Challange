#include <stdio.h>


int main() {
    float y = 69.89;

    int x = (int)(y); // explicit type conversion

    printf("Value of x: %d\n", x); // prints 20
    printf("Value of y: %f\n", y); // prints 20.500000

    char c = (char)x; // implicit type conversion

    printf("Value of c: %c\n", c); // prints 20

    


    return 0;
}
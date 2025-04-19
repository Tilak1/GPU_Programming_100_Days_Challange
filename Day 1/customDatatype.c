#include <stdio.h>


typedef struct {
    int x;
    int y;
} Point; // typedef is used to create an alias for a data type.

int main(){

Point p = {1,2};
printf("Point: (%d, %d)\n", p.x, p.y); // prints Point: (1, 2)

Point* ptr = &p; // pointer to struct
printf("Pointer to Point: (%d, %d)\n", ptr->x, ptr->y); // prints Pointer to Point: (1, 2)
printf("Pointer to Point: (%d, %d)\n", (*ptr).x, (*ptr).y); // prints Pointer to Point: (1, 2)



// ptr->x is used to access the members of the struct using the pointer
// ptr->y is used to access the members of the struct using the pointer
// ptr->x is equivalent to (*ptr).x
// ptr->y is equivalent to (*ptr).y
// (*ptr).x is used to access the members of the struct using the pointer
// (*ptr).y is used to access the members of the struct using the pointer

printf("Size of Point: %zu\n", sizeof(Point)); // prints Size of Point: 8 
// size of int + int 

}
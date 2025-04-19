// checking for null pointer avoids crashes 

#include <stdio.h>
#include <stdlib.h> // for malloc and free


int main(){

int* ptr = NULL; 

// NULL pointer is a pointer that does not point to any memory location
// It is used to indicate that the pointer is not initialized or does not point to a valid memory location
// It is a good practice to initialize pointers to NULL when they are declared

// It is used to avoid dereferencing a pointer that does not point to a valid memory location

// can be used when a malloc or new in c / cpp fails 

ptr = malloc(sizeof(int)); 
if (ptr==NULL)
{
    printf("Memory allocation failed\n");
    return 1; // exit the program with an error code
}
else
{
    printf("Memory allocation worked\n");
    return 0; // exit the program with an error code
}


return 0; // exit the program with success code
free(ptr); // free the allocated memory

}
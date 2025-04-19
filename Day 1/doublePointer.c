#include <stdio.h>

int main()
{
    int x = 10; 
    int* ptr = &x; 
    int** dptr = &ptr; // double pointer
    int*** tptr = &dptr;
    
    printf("The value of x is %d\n", ***tptr);

    return 0;
}
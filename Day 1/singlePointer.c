#include <stdio.h>

int main()
{
    int x = 10; 

    int* ptr = & x; 

    printf("The value of x is %d\n", *ptr); 
    printf("The address of x is %p\n", (void*)ptr);
    printf("The address of x is %p\n", (void*)&x);
    printf("The address of x is %p\n", &x);
        
    printf("Hello World");
    return 0;
}
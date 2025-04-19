#include <stdio.h>  

int main()
{
    int x = 10; 
    float y = 20; 

    void* ptr; 
    
    ptr = &x;

    // Derefferencing 
    // vptr is a memory address "&num" but it is stored as a void pointer (no data type)
    // We can't dereference a void pointer, so we cast it to an integer pointer to store the integer value at that memory address "(int*)vptr"
    // Then we dereference it with the final asterisk "*" to get the value "*((int*)vptr)"

    printf("Integer: %d\t\n",*(int*)ptr); // prints 10

    ptr = &y; 

    printf("Float: %f\t\n",*(float*)ptr); // prints 10

    return 0;
}
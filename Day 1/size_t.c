#include <stdio.h>

int main(){

int arr[] = {2,3,4,5,6}; 

size_t size = sizeof(arr)/sizeof(arr[0]); 
// size_t is an unsigned integer type that is used to represent the size of an object in bytes.
// It is used to store the size of an object in bytes.
// The sizeof operator returns the size of the object in bytes.

// %z for size_t & %u for unsigned int
// %zu for unsigned int for size_t


printf("Size of array is %zu\n", size); // %zu is used to print size_t type
printf("size of size_t type %zu\n",sizeof(size)); 
printf("size of long int %zu\n", sizeof( long unsigned int)); 
printf("size of int %zu\n", sizeof(long int)); 
printf("size of int %zu\n", sizeof(int));







}
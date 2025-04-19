#include <stdio.h>
#include <string.h>

// This program demonstrates the use of pointers with arrays in C.
// It initializes an array of integers and uses a pointer to iterate through the array,
// printing each element's value and address.
// The program also calculates the size of the array using the sizeof operator.
// The pointer is incremented to access each element of the array.


// we also get to see that the pointers are incrementted by 4 bytes at a time - becasue it is int / int32 / 32 bits / 4 bytes - type 
/*
The value of arr[0] is 32
The value of address for the elment arr[0] is 0x7ffd6cde8a40
The value of arr[1] is 15
The value of address for the elment arr[1] is 0x7ffd6cde8a44
The value of arr[2] is 12
The value of address for the elment arr[2] is 0x7ffd6cde8a48
The value of arr[3] is 34
The value of address for the elment arr[3] is 0x7ffd6cde8a4c
The value of arr[4] is 56
The value of address for the elment arr[4] is 0x7ffd6cde8a50
The value of arr[5] is 87
The value of address for the elment arr[5] is 0x7ffd6cde8a54
*/


int main()
{

int arr[]={32,15,12,34,56,87}; 
int* ptr = arr;

int size = sizeof(arr)/sizeof(arr)[0]; 


for (int i =0; i< (size); i++)
{
    printf("The value of arr[%d] is %d\n", i, *(ptr)); // dereferencing the pointer
    printf("The value of address for the elment arr[%d] is %p\n", i, (void*)ptr); // dereferencing the pointer
    
    ptr++; 

}
}
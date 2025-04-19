#include <stdio.h>
#include <stdlib.h> // for malloc and free

int main(){

    int arr [] = {1,2,3,4,5}; 
    int arr1 []=  {6,7,8,9,10}; 

    int* ptr = arr; 
    int* ptr1 = arr1; 
    
    int* matrix [] = {ptr,ptr1}; // 2D array

    for (int i=0;i<2;i++)
    {
        for (int j=0;j<5;j++)
        {
            printf("value for index i %d is %d \n",i, *matrix[i]++);
            //printf("address for index i %d is %p\n",i, (void*)matrix[i]++); // intial pointer is base and from base it keeps on incrementing 
        }
        printf("\n");
    }


    return 0; 
}
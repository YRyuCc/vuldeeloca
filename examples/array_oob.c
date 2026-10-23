#include <stdio.h>
#include <stdlib.h>

/* CWE-129: array index out of bounds */
int main(int argc, char **argv) {
    int arr[10];
    int idx = atoi(argv[1]);
    arr[idx] = 1;       /* vulnerable line: idx may exceed the array bounds */
    printf("%d\n", arr[idx]);
    return 0;
}

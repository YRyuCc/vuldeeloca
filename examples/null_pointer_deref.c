#include <stdio.h>
#include <stdlib.h>

/* CWE-476: null pointer dereference */
int main(int argc, char **argv) {
    int *p = NULL;
    if (argc > 10) {
        p = (int *)malloc(sizeof(int));
    }
    *p = 42;        /* vulnerable line: p may still be NULL */
    printf("%d\n", *p);
    return 0;
}

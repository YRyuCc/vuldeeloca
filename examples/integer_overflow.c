#include <stdio.h>
#include <limits.h>

/* CWE-190: signed integer overflow */
int main(void) {
    int a = INT_MAX;
    int b = a + 1;      /* vulnerable line: signed overflow is undefined */
    printf("%d\n", b);
    return 0;
}

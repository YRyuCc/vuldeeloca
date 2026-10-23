#include <stdio.h>
#include <string.h>

/* CWE-121: stack buffer overflow — strcat without a length check */
int main(int argc, char **argv) {
    char buf[16] = "hello";
    strcat(buf, argv[1]);   /* vulnerable line */
    printf("%s\n", buf);
    return 0;
}

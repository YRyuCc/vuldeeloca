#include <stdio.h>

/* CWE-134: uncontrolled format string — user data used as format */
int main(int argc, char **argv) {
    char *user = argv[1];
    printf(user);   /* vulnerable line */
    return 0;
}

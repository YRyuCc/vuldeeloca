from vuldee.source2slice.focus import extract_focus_points


SRC = """
#include <string.h>
void f(char *s) {
    char buf[10];
    int arr[8];
    int *p = arr;
    strcpy(buf, s);
    int x = arr[0] + 1;
}
"""


def test_extract_focus_points():
    points = extract_focus_points("test.c", SRC)
    types = {p.type.value for p in points}
    assert "FC" in types
    assert "AD" in types
    assert "PD" in types
    assert "AE" in types

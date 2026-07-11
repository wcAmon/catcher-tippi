#include "catcher.h"

int main(void) {
    catcher_destroy(NULL);
    return catcher_start(NULL) == CATCHER_INVALID_ARGUMENT ? 0 : 1;
}

#ifndef CATCHER_H
#define CATCHER_H

#include <stddef.h>
#include <stdint.h>

typedef struct catcher_handle catcher_handle_t;
typedef int32_t catcher_status_t;

enum {
    CATCHER_OK = 0,
    CATCHER_NO_UPDATE = 1,
    CATCHER_INVALID_ARGUMENT = -1,
    CATCHER_INVALID_STATE = -2,
    CATCHER_RUNTIME_ERROR = -3,
};

catcher_handle_t *catcher_create(const char *, const char *, uint32_t);
catcher_status_t catcher_start(catcher_handle_t *);
catcher_status_t catcher_push_audio(catcher_handle_t *, const float *, size_t);
catcher_status_t catcher_finish(catcher_handle_t *);
const char *catcher_text(const catcher_handle_t *);
const char *catcher_last_error(const catcher_handle_t *);
void catcher_destroy(catcher_handle_t *);

#endif

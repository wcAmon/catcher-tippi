#ifndef CATCHER_H
#define CATCHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct catcher_handle catcher_handle_t;
typedef int32_t catcher_status_t;

enum {
    CATCHER_OK = 0,
    CATCHER_NO_UPDATE = 1,
    CATCHER_INVALID_ARGUMENT = -1,
    CATCHER_INVALID_STATE = -2,
    CATCHER_RUNTIME_ERROR = -3,
};

catcher_handle_t *catcher_create(
    const char *model_path,
    const char *language,
    uint32_t lookahead
);

catcher_status_t catcher_start(catcher_handle_t *handle);
catcher_status_t catcher_push_audio(
    catcher_handle_t *handle,
    const float *samples,
    size_t count
);
catcher_status_t catcher_finish(catcher_handle_t *handle);
const char *catcher_text(const catcher_handle_t *handle);
const char *catcher_last_error(const catcher_handle_t *handle);
void catcher_destroy(catcher_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif

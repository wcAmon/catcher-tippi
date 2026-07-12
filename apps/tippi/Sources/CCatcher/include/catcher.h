#ifndef CATCHER_H
#define CATCHER_H

#include <stddef.h>
#include <stdint.h>

typedef struct catcher_handle catcher_handle_t;
typedef int32_t catcher_status_t;

/*
 * `catcher_push_audio`/`catcher_finish` return `CATCHER_OK` when new ASR
 * tokens arrived OR `catcher_segments` changed (diarization-only audio can
 * re-attribute/finalize a tentative segment with no new tokens);
 * `CATCHER_NO_UPDATE` means neither happened. Do not skip re-reading
 * `catcher_segments` on `CATCHER_NO_UPDATE` when a diarization model is
 * attached — see crates/catcher-ffi/include/catcher.h for the full contract.
 */
enum {
    CATCHER_OK = 0,
    CATCHER_NO_UPDATE = 1,
    CATCHER_INVALID_ARGUMENT = -1,
    CATCHER_INVALID_STATE = -2,
    CATCHER_RUNTIME_ERROR = -3,
};

/* Condensed mirror of crates/catcher-ffi/include/catcher.h — see that file
 * for the documented contract. `catcher_create`'s second argument is the
 * optional diarization model path (NULL = ASR only). */
catcher_handle_t *catcher_create(const char *, const char *, const char *, uint32_t);
catcher_status_t catcher_start(catcher_handle_t *);
catcher_status_t catcher_push_audio(catcher_handle_t *, const float *, size_t);
catcher_status_t catcher_finish(catcher_handle_t *);
const char *catcher_text(const catcher_handle_t *);
const char *catcher_segments(const catcher_handle_t *);
const char *catcher_warning(const catcher_handle_t *);
const char *catcher_last_error(const catcher_handle_t *);
void catcher_destroy(catcher_handle_t *);

#endif

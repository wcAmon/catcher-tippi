#ifndef CATCHER_H
#define CATCHER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct catcher_handle catcher_handle_t;
typedef int32_t catcher_status_t;

/*
 * `catcher_push_audio`/`catcher_finish` return `CATCHER_OK` when either new
 * ASR tokens were decoded from the call *or* `catcher_segments` came out
 * different than it was beforehand; they return `CATCHER_NO_UPDATE` only
 * when neither happened. A diarization-only chunk of audio (no new ASR
 * tokens at all) can still re-attribute or finalize a tentative trailing
 * segment, so `CATCHER_NO_UPDATE` does NOT imply "catcher_segments is
 * unchanged" was already true before diarization was added — callers must
 * re-read `catcher_segments` on every call regardless of status once a
 * diarization model is attached. `catcher_finish` in particular forces every
 * trailing segment final, so it returns `CATCHER_OK` in almost every
 * diarization-enabled case. For an ASR-only handle (NULL `diar_model_path`),
 * `catcher_segments` never changes, so `CATCHER_NO_UPDATE` keeps its original
 * v1 meaning: no new transcript text.
 */
enum {
    CATCHER_OK = 0,
    CATCHER_NO_UPDATE = 1,
    CATCHER_INVALID_ARGUMENT = -1,
    CATCHER_INVALID_STATE = -2,
    CATCHER_RUNTIME_ERROR = -3,
};

/*
 * Loads a Catcher ASR model and, if `diar_model_path` is non-null, a
 * Sortformer diarization model, creating an idle handle.
 *
 * `diar_model_path` is optional: pass NULL for ASR-only transcription
 * (`catcher_segments` then always returns "[]" and `catcher_warning` always
 * returns NULL). When `diar_model_path` is non-null and fails to load,
 * `catcher_create` fails outright (returns NULL; see `catcher_last_error`)
 * exactly as an `asr_model_path` load failure would. Once a diarization
 * model is loaded, a *runtime* diarization failure during
 * `catcher_push_audio`/`catcher_finish` instead degrades gracefully: it is
 * reported via `catcher_warning` and transcription continues unaffected.
 */
catcher_handle_t *catcher_create(
    const char *asr_model_path,
    const char *diar_model_path,
    const char *language,
    uint32_t lookahead
);

/*
 * Clears caches and text for a new utterance. If the diarizer previously
 * degraded to disabled after a runtime error (see catcher_warning),
 * catcher_start attempts to rebuild it in place from the same
 * `diar_model_path` given to catcher_create. A successful rebuild clears the
 * warning and resumes diarization; a failed rebuild is non-fatal: it still
 * returns CATCHER_OK, leaves a warning describing the reload failure (see
 * catcher_warning), and the handle continues in ASR-only mode for the new
 * utterance. Never fails outright for this reason alone.
 */
catcher_status_t catcher_start(catcher_handle_t *handle);
catcher_status_t catcher_push_audio(
    catcher_handle_t *handle,
    const float *samples,
    size_t count
);
catcher_status_t catcher_finish(catcher_handle_t *handle);
catcher_status_t catcher_finish_before(catcher_handle_t *handle, uint64_t cutoff_ms);

/*
 * Returns the current UTF-8 transcript owned by `handle`. The pointer is
 * borrowed: it stays valid until the next mutating call on this handle
 * (catcher_start/catcher_push_audio/catcher_finish) or catcher_destroy,
 * whichever happens first. Returns NULL if `handle` is NULL.
 */
const char *catcher_text(const catcher_handle_t *handle);

/*
 * Returns the current speaker segments as a UTF-8 JSON array, e.g.
 * `[{"speaker":0,"start_ms":0,"end_ms":800,"text":"...","final":true}]`.
 * `"final":false` marks at most one trailing tentative segment whose
 * speaker attribution may still change as more audio/diarization frames
 * arrive; call catcher_finish to force every segment final.
 *
 * Returns the literal `"[]"` when `catcher_create` was called with a NULL
 * `diar_model_path` (no diarization model was ever loaded for this handle).
 *
 * Same borrowed-pointer lifetime rules as catcher_text: valid until the next
 * mutating call on this handle or catcher_destroy. Returns NULL if `handle`
 * is NULL.
 */
const char *catcher_segments(const catcher_handle_t *handle);

/*
 * Returns a non-fatal diarization warning, or NULL when there is nothing to
 * report. Set once a loaded diarization model hits a runtime error during
 * catcher_push_audio/catcher_finish; reads "diarization disabled after a
 * runtime error: <details>". The diarizer stays disabled for the rest of
 * the utterance (segments then keep flowing from whatever diarization
 * frames had already arrived, without new ones). catcher_start clears the
 * warning and attempts to rebuild the diarizer in place; if that rebuild
 * fails, a new warning is set instead, reading "diarization unavailable:
 * failed to reload the model: <details>", and the handle stays in ASR-only
 * mode for the new utterance. Transcription itself is unaffected by any
 * condition reported here.
 *
 * Same borrowed-pointer lifetime rules as catcher_text: valid until the next
 * mutating call on this handle or catcher_destroy. Returns NULL if `handle`
 * is NULL (indistinguishable from "no warning"; check catcher_last_error to
 * tell the two apart).
 */
const char *catcher_warning(const catcher_handle_t *handle);

const char *catcher_last_error(const catcher_handle_t *handle);
void catcher_destroy(catcher_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif

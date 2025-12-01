// Stub implementations for tch-rs stream callbacks
// These are normally provided by Rust, but we stub them out for Idris FFI

#include <cstddef>

extern "C" {

void tch_read_stream_destructor(void *stream_ptr) {
    // No-op: stream cleanup handled by Idris
}

size_t tch_read_stream_read(void *stream_ptr, char *buf, size_t len) {
    // Stub: return 0 bytes read
    return 0;
}

void tch_read_stream_seek_start(void *stream_ptr, size_t pos) {
    // No-op stub
}

void tch_read_stream_seek_end(void *stream_ptr) {
    // No-op stub
}

size_t tch_read_stream_stream_position(void *stream_ptr) {
    // Stub: return 0
    return 0;
}

void tch_write_stream_destructor(void *stream_ptr) {
    // No-op: stream cleanup handled by Idris
}

size_t tch_write_stream_write(void *stream_ptr, const char *buf, size_t len) {
    // Stub: return 0 bytes written
    return 0;
}

} // extern "C"

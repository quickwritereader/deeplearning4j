package org.nd4j.aurora;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

public class Aurora extends AuroraPresets {
    static { Loader.load(); }

// Parsed from ve_offload.h

    /* Copyright (C) 2017-2020 by NEC Corporation
     *
     * Permission is hereby granted, free of charge, to any person obtaining a copy
     * of this software and associated documentation files (the "Software"), to
     * deal in the Software without restriction, including without limitation the
     * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
     * sell copies of the Software, and to permit persons to whom the Software is
     * furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in
     * all copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
     * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
     * IN THE SOFTWARE.
     */
    /**
     * \file ve_offload.h
     */
// #ifndef _VE_OFFLOAD_H_
// #define _VE_OFFLOAD_H_

    public static final int VEO_API_VERSION = 9;
    public static final int VEO_SYMNAME_LEN_MAX = (255);
    public static final String VEO_LOG_CATEGORY = "veos.veo.veo";
    public static final int VEO_MAX_NUM_ARGS = (256);

    public static final long VEO_REQUEST_ID_INVALID = (~0L);

// #include <stdint.h>
// #include <stddef.h>

// #ifdef __cplusplus
// #endif
    /** enum veo_context_state */
    public static final int
            VEO_STATE_UNKNOWN = 0,
            VEO_STATE_RUNNING = 1,
            VEO_STATE_SYSCALL = 2,	// not possible in AVEO
            VEO_STATE_BLOCKED = 3,	// not possible in AVEO
            VEO_STATE_EXIT = 4;

    /** enum veo_command_state */
    public static final int
            VEO_COMMAND_OK = 0,
            VEO_COMMAND_EXCEPTION = 1,
            VEO_COMMAND_ERROR = 2,
            VEO_COMMAND_UNFINISHED = 3;

    /** enum veo_queue_state */
    public static final int
            VEO_QUEUE_READY = 0,
            VEO_QUEUE_CLOSED = 1;

    /** enum veo_args_intent */
    public static final int
            VEO_INTENT_IN = 0,
            VEO_INTENT_INOUT = 1,
            VEO_INTENT_OUT = 2;

    @Opaque public static class veo_args extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public veo_args() { super((Pointer)null); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public veo_args(Pointer p) { super(p); }
    }
    @Opaque public static class veo_proc_handle extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public veo_proc_handle() { super((Pointer)null); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public veo_proc_handle(Pointer p) { super(p); }
    }
    @Opaque public static class veo_thr_ctxt extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public veo_thr_ctxt() { super((Pointer)null); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public veo_thr_ctxt(Pointer p) { super(p); }
    }
    @Opaque public static class veo_thr_ctxt_attr extends Pointer {
        /** Empty constructor. Calls {@code super((Pointer)null)}. */
        public veo_thr_ctxt_attr() { super((Pointer)null); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public veo_thr_ctxt_attr(Pointer p) { super(p); }
    }

    //struct veo_proc_handle *veo_proc_create(int);
    @NoException public static native veo_proc_handle veo_proc_create(int arg0);
    @NoException public static native veo_proc_handle veo_proc_create_static(int arg0, @Cast("char*") BytePointer arg1);
    @NoException public static native veo_proc_handle veo_proc_create_static(int arg0, @Cast("char*") ByteBuffer arg1);
    @NoException public static native veo_proc_handle veo_proc_create_static(int arg0, @Cast("char*") byte[] arg1);
    @NoException public static native veo_proc_handle veo_proc_create_static(int arg0, @Cast("char*") String arg1);
    @NoException public static native int veo_proc_destroy(veo_proc_handle arg0);
    @NoException public static native @Cast("uint64_t") long veo_load_library(veo_proc_handle arg0, @Cast("const char*") BytePointer arg1);
    @NoException public static native @Cast("uint64_t") long veo_load_library(veo_proc_handle arg0, @Cast("const char*") ByteBuffer arg1);
    @NoException public static native @Cast("uint64_t") long veo_load_library(veo_proc_handle arg0, @Cast("const char*") byte[] arg1);
    @NoException public static native @Cast("uint64_t") long veo_load_library(veo_proc_handle arg0, @Cast("const char*") String arg1);
    @NoException public static native int veo_unload_library(veo_proc_handle arg0, @Cast("const uint64_t") long arg1);
    @NoException public static native @Cast("uint64_t") long veo_get_sym(veo_proc_handle arg0, @Cast("uint64_t") long arg1, @Cast("const char*") BytePointer arg2);
    @NoException public static native @Cast("uint64_t") long veo_get_sym(veo_proc_handle arg0, @Cast("uint64_t") long arg1, @Cast("const char*") ByteBuffer arg2);
    @NoException public static native @Cast("uint64_t") long veo_get_sym(veo_proc_handle arg0, @Cast("uint64_t") long arg1, @Cast("const char*") byte[] arg2);
    @NoException public static native @Cast("uint64_t") long veo_get_sym(veo_proc_handle arg0, @Cast("uint64_t") long arg1, @Cast("const char*") String arg2);
    @NoException public static native int veo_alloc_mem(veo_proc_handle arg0, @Cast("uint64_t*") LongPointer arg1, @Cast("const size_t") long arg2);
    @NoException public static native int veo_alloc_mem(veo_proc_handle arg0, @Cast("uint64_t*") LongBuffer arg1, @Cast("const size_t") long arg2);
    @NoException public static native int veo_alloc_mem(veo_proc_handle arg0, @Cast("uint64_t*") long[] arg1, @Cast("const size_t") long arg2);
    @NoException public static native int veo_free_mem(veo_proc_handle arg0, @Cast("uint64_t") long arg1);
    @NoException public static native int veo_read_mem(veo_proc_handle arg0, Pointer arg1, @Cast("uint64_t") long arg2, @Cast("size_t") long arg3);
    @NoException public static native int veo_write_mem(veo_proc_handle arg0, @Cast("uint64_t") long arg1, @Const Pointer arg2, @Cast("size_t") long arg3);
    @NoException public static native int veo_num_contexts(veo_proc_handle arg0);
    @NoException public static native veo_thr_ctxt veo_get_context(veo_proc_handle arg0, int arg1);

    @NoException public static native veo_thr_ctxt veo_context_open(veo_proc_handle arg0);
    @NoException public static native int veo_context_close(veo_thr_ctxt arg0);
    @NoException public static native int veo_get_context_state(veo_thr_ctxt arg0);
    @NoException public static native void veo_context_sync(veo_thr_ctxt arg0);

    @NoException public static native veo_args veo_args_alloc();
    @NoException public static native int veo_args_set_i64(veo_args arg0, int arg1, @Cast("int64_t") long arg2);
    @NoException public static native int veo_args_set_u64(veo_args arg0, int arg1, @Cast("uint64_t") long arg2);
    @NoException public static native int veo_args_set_i32(veo_args arg0, int arg1, int arg2);
    @NoException public static native int veo_args_set_u32(veo_args arg0, int arg1, @Cast("uint32_t") int arg2);
    @NoException public static native int veo_args_set_i16(veo_args arg0, int arg1, short arg2);
    @NoException public static native int veo_args_set_u16(veo_args arg0, int arg1, @Cast("uint16_t") short arg2);
    @NoException public static native int veo_args_set_i8(veo_args arg0, int arg1, byte arg2);
    @NoException public static native int veo_args_set_u8(veo_args arg0, int arg1, @Cast("uint8_t") byte arg2);
    @NoException public static native int veo_args_set_double(veo_args arg0, int arg1, double arg2);
    @NoException public static native int veo_args_set_float(veo_args arg0, int arg1, float arg2);
    @NoException public static native int veo_args_set_stack(veo_args arg0, @Cast("veo_args_intent") int arg1,
                                                             int arg2, @Cast("char*") BytePointer arg3, @Cast("size_t") long arg4);
    @NoException public static native int veo_args_set_stack(veo_args arg0, @Cast("veo_args_intent") int arg1,
                                                             int arg2, @Cast("char*") ByteBuffer arg3, @Cast("size_t") long arg4);
    @NoException public static native int veo_args_set_stack(veo_args arg0, @Cast("veo_args_intent") int arg1,
                                                             int arg2, @Cast("char*") byte[] arg3, @Cast("size_t") long arg4);
    @NoException public static native int veo_args_set_stack(veo_args arg0, @Cast("veo_args_intent") int arg1,
                                                             int arg2, @Cast("char*") String arg3, @Cast("size_t") long arg4);
    @NoException public static native void veo_args_clear(veo_args arg0);
    @NoException public static native void veo_args_free(veo_args arg0);

    @NoException public static native int veo_call_sync(veo_proc_handle h, @Cast("uint64_t") long addr,
                                                        veo_args ca, @Cast("uint64_t*") LongPointer result);
    @NoException public static native int veo_call_sync(veo_proc_handle h, @Cast("uint64_t") long addr,
                                                        veo_args ca, @Cast("uint64_t*") LongBuffer result);
    @NoException public static native int veo_call_sync(veo_proc_handle h, @Cast("uint64_t") long addr,
                                                        veo_args ca, @Cast("uint64_t*") long[] result);

    @NoException public static native @Cast("uint64_t") long veo_call_async(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, veo_args arg2);
    @NoException public static native @Cast("uint64_t") long veo_call_async_by_name(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("const char*") BytePointer arg2, veo_args arg3);
    @NoException public static native @Cast("uint64_t") long veo_call_async_by_name(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("const char*") ByteBuffer arg2, veo_args arg3);
    @NoException public static native @Cast("uint64_t") long veo_call_async_by_name(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("const char*") byte[] arg2, veo_args arg3);
    @NoException public static native @Cast("uint64_t") long veo_call_async_by_name(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("const char*") String arg2, veo_args arg3);
    public static class Arg1_Pointer extends FunctionPointer {
        static { Loader.load(); }
        /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
        public    Arg1_Pointer(Pointer p) { super(p); }
        protected Arg1_Pointer() { allocate(); }
        private native void allocate();
        public native @Cast("uint64_t") long call(Pointer arg0);
    }
    @NoException public static native @Cast("uint64_t") long veo_call_async_vh(veo_thr_ctxt arg0, Arg1_Pointer arg1, Pointer arg2);

    @NoException public static native int veo_call_peek_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") LongPointer arg2);
    @NoException public static native int veo_call_peek_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") LongBuffer arg2);
    @NoException public static native int veo_call_peek_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") long[] arg2);
    @NoException public static native int veo_call_wait_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") LongPointer arg2);
    @NoException public static native int veo_call_wait_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") LongBuffer arg2);
    @NoException public static native int veo_call_wait_result(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Cast("uint64_t*") long[] arg2);

    @NoException public static native @Cast("uint64_t") long veo_async_read_mem(veo_thr_ctxt arg0, Pointer arg1, @Cast("uint64_t") long arg2, @Cast("size_t") long arg3);
    @NoException public static native @Cast("uint64_t") long veo_async_write_mem(veo_thr_ctxt arg0, @Cast("uint64_t") long arg1, @Const Pointer arg2,
                                                                                 @Cast("size_t") long arg3);

    @NoException public static native veo_thr_ctxt veo_context_open_with_attr(
            veo_proc_handle arg0, veo_thr_ctxt_attr arg1);
    @NoException public static native veo_thr_ctxt_attr veo_alloc_thr_ctxt_attr();
    @NoException public static native int veo_free_thr_ctxt_attr(veo_thr_ctxt_attr arg0);
    @NoException public static native int veo_set_thr_ctxt_stacksize(veo_thr_ctxt_attr arg0, @Cast("size_t") long arg1);
    @NoException public static native int veo_get_thr_ctxt_stacksize(veo_thr_ctxt_attr arg0, @Cast("size_t*") SizeTPointer arg1);

    @NoException public static native @Cast("const char*") BytePointer veo_version_string();
    @NoException public static native int veo_api_version();

// #ifdef __cplusplus // extern "C"
// #endif
// #endif


}
#include "include/libcuda_hook.h"
#include "multiprocess/multiprocess_memory_limit.h"
#include "multiprocess/multiprocess_utilization_watcher.h"

CUresult cuStreamCreate(CUstream *phstream, unsigned int flags){
    LOG_INFO("cuStreamCreate %p",phstream);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamCreate,phstream,flags);
    return res;
}

CUresult cuStreamDestroy_v2 ( CUstream hStream ){
    LOG_DEBUG("cuStreamDestroy_v2 %p",hStream);
    return CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamDestroy_v2,hStream);
}

CUresult cuStreamSynchronize(CUstream hstream){
    LOG_DEBUG("cuStreamSync %p",hstream);
    CUresult res = CUDA_OVERRIDE_CALL(cuda_library_entry,cuStreamSynchronize,hstream);
    if (res == CUDA_SUCCESS && get_time_based_throttle()) {
        CUdevice current_device;
        int device_id = (cuCtxGetDevice(&current_device) == CUDA_SUCCESS) ? (int)current_device : 0;
        time_throttle_sync(device_id);
    }
    return res;
}

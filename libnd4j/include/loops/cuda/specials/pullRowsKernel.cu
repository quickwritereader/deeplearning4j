/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma, created on 15.11.2018
//
#include <loops/special_kernels.h>

namespace sd {

///////////////////////////////////////////////////////////////////////
    template<typename T>
    SD_DEVICE void pullRowsKernel(void *vx,
                                   void *vz,
                                   sd::LongType len,
                                   sd::LongType *indexes,
                                   sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                   sd::LongType const* zTadShapeInfo, sd::LongType const* zTadOffsets) {

        auto x = reinterpret_cast<T *>(vx);
        auto z = reinterpret_cast<T *>(vz);
        auto xEWS = shape::elementWiseStride(tadShapeInfo);
        auto zEWS = shape::elementWiseStride(zTadShapeInfo);
        auto tadLength = shape::length(tadShapeInfo);

        if (xEWS >= 1 && zEWS >= 1) {
            for (int idx = blockIdx.x; idx < len; idx += gridDim.x) {
                T *rX = x + tadOffsets[indexes[idx]];
                T *rZ = z + zTadOffsets[idx];

                for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                    rZ[i * zEWS] = rX[i * xEWS];
                }
            }
        } else {
            for (int idx = blockIdx.x; idx < len; idx += gridDim.x) {
                T *rX = x + tadOffsets[indexes[idx]];
                T *rZ = z + zTadOffsets[idx];

                for (int i = threadIdx.x; i < tadLength; i += blockDim.x) {
                    auto xOffset = shape::getIndexOffset(i, tadShapeInfo);
                    auto zOffset = shape::getIndexOffset(i, zTadShapeInfo);
                    rZ[zOffset] = rX[xOffset];
                }
            }
        }
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    SD_KERNEL void execPullRowsKernel(void *vx,
                                       void *vz,
                                       sd::LongType len,
                                       sd::LongType *indexes,
                                       sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                       sd::LongType const* zTadShapeInfo, sd::LongType const* zTadOffsets) {

        pullRowsKernel<T>(vx, vz, len, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
    }

///////////////////////////////////////////////////////////////////////
    template<typename T>
    SD_HOST void pullRowsKernelGeneric(dim3 &launchDims, cudaStream_t *stream,
                                        void *vx,
                                        void *vz,
                                        sd::LongType len,
                                        sd::LongType *indexes,
                                        sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets,
                                        sd::LongType const* zTadShapeInfo, sd::LongType const* zTadOffsets) {

        execPullRowsKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(vx, vz, len, indexes, tadShapeInfo, tadOffsets, zTadShapeInfo, zTadOffsets);
        sd::DebugHelper::checkErrorCode(stream, "pullRows(...) failed");
    }

    BUILD_SINGLE_TEMPLATE(template void SD_LIB_HIDDEN pullRowsKernelGeneric, (dim3 & launchDims, cudaStream_t * stream, void * vx, void * vz, sd::LongType len, sd::LongType * indexes, sd::LongType const* tadShapeInfo, sd::LongType const* tadOffsets, sd::LongType const* zTadShapeInfo, sd::LongType const* zTadOffsets), SD_COMMON_TYPES);
}


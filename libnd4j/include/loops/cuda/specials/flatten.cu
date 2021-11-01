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
// @author Yurii Shyrma, created on 27.11.2018
//
#include <loops/special_kernels.h>
#include <ops/declarable/helpers/flatten.h>

namespace sd {

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_KERNEL void flattenKernel(
                    sd::Pointer *extraPointers,
                    int dOffset,
                    char order,
                    void *vz, sd::LongType *zShapeInfo,
                    void *vy, sd::LongType *yShapeInfo) {

    auto z = reinterpret_cast<T*>(vz);
    auto y = reinterpret_cast<T*>(vy);

    __shared__ sd::LongType lenY, yOrder, zEWS, yEWS;

    if (threadIdx.x == 0) {

        yEWS = shape::elementWiseStride(yShapeInfo);
        zEWS = shape::elementWiseStride(zShapeInfo);
        lenY = shape::length(yShapeInfo);
    }
    __syncthreads();

    sd::LongType tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(auto i = tid; i < lenY; i += gridDim.x * blockDim.x)
        z[i * zEWS + dOffset] = y[ops::helpers::getIndexOffsetOrdered(i, yShapeInfo, order)];
}

////////////////////////////////////////////////////////////////////////
template <typename T>
SD_HOST void flattenKernelGeneric(dim3& launchDims, cudaStream_t *stream,
                            sd::Pointer *extraPointers,
                            int dOffset,
                            char order,
                            void *vz, sd::LongType *zShapeInfo,
                            void *vy, sd::LongType *yShapeInfo) {

    flattenKernel<T><<<launchDims.x, launchDims.y, launchDims.z, *stream>>>(extraPointers, dOffset, order, vz, zShapeInfo, vy, yShapeInfo);
    sd::DebugHelper::checkErrorCode(stream, "flattenGeneric(...) failed");
}

BUILD_SINGLE_TEMPLATE(template void SD_LIB_HIDDEN flattenKernelGeneric, (dim3& launchDims, cudaStream_t *stream, sd::Pointer *extraPointers, int dOffset, char order, void *vz, sd::LongType *zShapeInfo, void *vy, sd::LongType *yShapeInfo), SD_COMMON_TYPES);


}
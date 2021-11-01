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

/*
 * pairwise_transform.h
 *
 *  Created on: Dec 28, 2015
 *      Author: agibsonccc
 */

#ifndef PAIRWISE_TRANSFORM_H_
#define PAIRWISE_TRANSFORM_H_


#include <stdio.h>
#include <ops/ops.h>
#include <system/op_boilerplate.h>
#include <types/types.h>
#include <loops/legacy_ops.h>
#include <helpers/DebugHelper.h>

#ifdef __CUDACC__
#include <cuda.h>
#include <cuda_runtime.h>
#include <types/float16.h>
#endif


namespace functions {
    namespace pairwise_transforms {

/**
 * Transforms involving 2 arrays
 */
        template<typename X, typename Y, typename Z>
        class PairWiseTransform {
        public:

#ifdef __CUDABLAS__

            template <typename OpType>            
            static SD_HOST void intermediateShaped(dim3& launchDims, cudaStream_t *stream,
                                                    const void *vx, const sd::LongType *xShapeInfo,
                                                    const void *vy, const sd::LongType *yShapeInfo,
                                                    void *vz, const sd::LongType *zShapeInfo,
                                                    void *vextraParams);

            static SD_HOST void executeCudaShaped(dim3& launchDims, cudaStream_t *stream,
                                                   int opNum,
                                                   const void *x, const sd::LongType *xShapeInfo,
                                                   const void *y, const sd::LongType *yShapeInfo,
                                                   void *z, const sd::LongType *zShapeInfo,
                                                   void *extraParams);

#endif
        public:

            static void exec(int opNum,
                             const void *x, const sd::LongType *xShapeInfo,
                             const void *y, const sd::LongType *yShapeInfo,
                             void *z, const sd::LongType *zShapeInfo,
                             void *extraParams,
                             uint64_t start, uint64_t stop);

            static void exec(int opNum,
                             const void *x, sd::LongType xStride,
                             const void *y, sd::LongType yStride,
                             void *z, sd::LongType resultStride,
                             void *extraParams,
                             sd::LongType len,
                             uint64_t start, uint64_t stop);


            template<typename OpType>
            static void exec(const void *vx, const sd::LongType* xShapeInfo,
                             const void *vy, const sd::LongType* yShapeInfo,
                             void *vresult, const sd::LongType* zShapeInfo,
                             void *vextraParams,
                             uint64_t start, uint64_t stop);

            template<typename OpType>
            static void exec(const void *vx, sd::LongType xStride,
                             const void *vy, sd::LongType yStride,
                             void *vresult, sd::LongType resultStride,
                             void *vextraParams,
                             sd::LongType len,
                             uint64_t start, uint64_t stop);
        };
    }
}

#endif /* PAIRWISE_TRANSFORM_H_ */

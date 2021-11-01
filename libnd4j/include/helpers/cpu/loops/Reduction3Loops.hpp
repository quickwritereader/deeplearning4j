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
//
#include <helpers/Loops.h>
#include <types/types.h>

using namespace simdOps;

namespace sd {

    template<typename X, typename Z>
    template <typename OpType>
    SD_LIB_HIDDEN void Reduction3Loops<X,Z>::innerloopReduce3(const X* x, const sd::LongType* xShapeInfo, const X* y, const sd::LongType* yShapeInfo, Z* z, const sd::LongType* zShapeInfo, int* dims, int dimsLen, Z* extraParams, int64_t start, int64_t stop) {
#ifndef SD_LOOPS_INLINED
        Reduction3Loops<X,Z>::template loopReduce3<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dims, dimsLen, extraParams, start, stop);
#endif
    }

    template<typename X, typename Z>
    template <typename OpType>
    SD_LIB_HIDDEN void Reduction3Loops<X,Z>::innerloopReduce3All(const X* x, const sd::LongType* xShapeInfo, const X* y, const sd::LongType* yShapeInfo, Z* z, const sd::LongType* zShapeInfo, const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffsets, const sd::LongType* yTadShapeInfo, const sd::LongType* yTadOffsets, Z* extraParams, int64_t start, int64_t stop) {
#ifndef SD_LOOPS_INLINED
        Reduction3Loops<X,Z>::template loopReduce3All<OpType>(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, xTadShapeInfo, xTadOffsets, yTadShapeInfo, yTadOffsets, extraParams, start, stop);
#endif
    }

    template<typename X, typename Y>
    SD_LIB_HIDDEN void Reduction3Loops<X, Y>::wrapper(const int opNum, const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Y *z, const sd::LongType *zShapeInfo, int* dims, int dimsLen, Y *extraParams, int64_t start, int64_t stop) {
#ifndef SD_LOOPS_INLINED
        DISPATCH_BY_OPNUM_TT(innerloopReduce3, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo, dims, dimsLen, extraParams, start, stop), REDUCE3_OPS);
#endif
    }

    template<typename X, typename Y>
    SD_LIB_HIDDEN void Reduction3Loops<X, Y>::wrapperAll(const int opNum, const X *x, const sd::LongType *xShapeInfo, const X *y, const sd::LongType *yShapeInfo, Y *z, const sd::LongType *zShapeInfo, const sd::LongType* xTadShapeInfo, const sd::LongType* xTadOffsets, const sd::LongType* yTadShapeInfo, const sd::LongType* yTadOffsets, Y* extraParams, int64_t start, int64_t stop) {
#ifndef SD_LOOPS_INLINED
        DISPATCH_BY_OPNUM_TT(innerloopReduce3All, PARAMS(x, xShapeInfo, y, yShapeInfo, z, zShapeInfo,  xTadShapeInfo, xTadOffsets, yTadShapeInfo, yTadOffsets, extraParams, start, stop), REDUCE3_OPS);
#endif
    }
     
}

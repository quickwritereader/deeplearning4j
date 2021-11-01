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
// @author Oleh Semeniv (oleg.semeniv@gmail.com)
//
#include <system/op_boilerplate.h>
#include <ops/declarable/helpers/updatersHelpers.h>
#include <helpers/PointersManager.h>
#include <math/platformmath.h>
#include <math/templatemath.h>

namespace sd    {
namespace ops     {
namespace helpers {

///////////////////////////////////////////////////////////////////
template<typename T>
SD_KERNEL void adaMaxUpdaterCuda(const void* vx, const sd::LongType* xShapeInfo, const void* vinv, const sd::LongType* invShapeInfo, 
    const void* vinm, const sd::LongType* inmShapeInfo,  void* vz, const sd::LongType* zShapeInfo, 
    void* vstV, const sd::LongType* stvShapeInfo, void* vstM, const sd::LongType* stmShapeInfo, 
    const T lr, const T beta1, const T beta2, const T epsilon, const T iteration) {

    const auto grad = reinterpret_cast<const T*>(vx);
    const auto initU = reinterpret_cast<const T*>(vinv);
    const auto initM = reinterpret_cast<const T*>(vinm);
    
    auto up = reinterpret_cast<T*>(vz);
    auto stU = reinterpret_cast<T*>(vstV);
    auto stM = reinterpret_cast<T*>(vstM);

    __shared__ sd::LongType xLen;
    __shared__ T beta1T, epsilonT;
    __shared__ bool bEWS, bOrdering, bXZsame, bXInUSame, bXStUSame, bXInMSame, bXStMSame;

    if (threadIdx.x == 0) {
        xLen = shape::length(xShapeInfo);
        beta1T = sd::math::sd_pow<T,T,T>(beta1, (iteration + 1) );

        epsilonT = lr / (1.0 - beta1T);
        if (sd::math::sd_isnan(epsilonT) || 0 == epsilonT || sd::math::sd_isinf(epsilonT))
            epsilonT = epsilon;

        bEWS =  1 == shape::elementWiseStride(xShapeInfo) && 1 == shape::elementWiseStride(zShapeInfo) &&
                1 == shape::elementWiseStride(stmShapeInfo) && 1 == shape::elementWiseStride(inmShapeInfo) &&
                1 == shape::elementWiseStride(stvShapeInfo) && 1 == shape::elementWiseStride(invShapeInfo);
        bOrdering = shape::order(xShapeInfo) == shape::order(zShapeInfo) && shape::order(xShapeInfo) == shape::order(stmShapeInfo) &&
                    shape::order(xShapeInfo) == shape::order(inmShapeInfo) && shape::order(xShapeInfo) == shape::order(invShapeInfo) &&
                    shape::order(xShapeInfo) == shape::order(stvShapeInfo);

        bXZsame = shape::haveSameShapeAndStrides(xShapeInfo, zShapeInfo);
        bXInUSame = shape::haveSameShapeAndStrides(xShapeInfo, invShapeInfo);
        bXStUSame = shape::haveSameShapeAndStrides(xShapeInfo, stvShapeInfo);
        bXInMSame = shape::haveSameShapeAndStrides(xShapeInfo, inmShapeInfo);
        bXStMSame = shape::haveSameShapeAndStrides(xShapeInfo, stmShapeInfo);
    }
    __syncthreads();

    int coords[SD_MAX_RANK];

    for (sd::LongType i = blockIdx.x * blockDim.x + threadIdx.x; i < xLen; i += gridDim.x * blockDim.x) {
        

        auto xOffset = i, zOffset = i, initMOffset = i, initUOffset = i, stMOffset = i, stUOffset = i;

        if (!bEWS || !bOrdering) {

            shape::index2coords(i, xShapeInfo, coords);
            xOffset  = shape::getOffset(xShapeInfo, coords);
            zOffset  = bXZsame ? xOffset : shape::getOffset(zShapeInfo, coords);
            initUOffset = bXInUSame ? xOffset : shape::getOffset(invShapeInfo, coords);
            stUOffset = bXStUSame ? xOffset : shape::getOffset(stvShapeInfo, coords);
            initMOffset = bXInMSame ? xOffset : shape::getOffset(inmShapeInfo, coords);
            stMOffset = bXStMSame ? xOffset : shape::getOffset(stmShapeInfo, coords);
        }

        //m = B_1 * m + (1-B_1)*grad
        stM[stMOffset] = beta1 * initM[initMOffset] + grad[xOffset] * (1 - beta1);
        //u = max(B_2 * u, |grad|)
        stU[stUOffset] = sd::math::sd_max( (beta2* initU[initUOffset]), sd::math::sd_abs(grad[xOffset])) + 1e-32;
        
        up[zOffset] = (stM[stMOffset] * epsilonT) / stU[stUOffset];
    }
}

///////////////////////////////////////////////////////////////////
template<typename T>
void adaMaxUpdaterCudaLauncher(const int blocksPerGrid, const int threadsPerBlock, const cudaStream_t* stream, const void* vx, const sd::LongType* xShapeInfo, 
                                       const void* vinv, const sd::LongType* invShapeInfo, const void* vinm, const sd::LongType* inmShapeInfo, 
                                        void* vz, const sd::LongType* zShapeInfo, void* vstV, const sd::LongType* stvShapeInfo, 
                                        void* vstM, const sd::LongType* stmShapeInfo, const double dLr, 
                                        const double dBeta1, const double dBeta2, const double dEpsilon, const int nIteration) {
    
    const T lr = static_cast<T>(dLr);
    const T beta1 = static_cast<T>(dBeta1);
    const T beta2 = static_cast<T>(dBeta2);
    const T epsilon = static_cast<T>(dEpsilon);
    const T iteration = static_cast<T>(nIteration);

    adaMaxUpdaterCuda<T><<<blocksPerGrid, threadsPerBlock, 256, * stream>>>(vx, xShapeInfo, vinv, invShapeInfo, vinm, inmShapeInfo, vz,
         zShapeInfo, vstV, stvShapeInfo, vstM, stmShapeInfo, lr, beta1, beta2, epsilon, iteration);
}

///////////////////////////////////////////////////////////////////
void updaterAdaMax(sd::LaunchContext* context, const NDArray& gradient, const NDArray& initStateU, const NDArray& initStateM,
                   NDArray& update, NDArray& stateU, NDArray& stateM, const double dLr, const double dBeta1, 
                   const double dBeta2, const double dEpsilon, const int nIteration) {

    PointersManager manager(context, "adaMaxUpdater");

    const int threadsPerBlock = SD_MAX_NUM_THREADS / 4;
    const int blocksPerGrid = (gradient.lengthOf() + threadsPerBlock - 1) / threadsPerBlock;

    NDArray::prepareSpecialUse({ &update, &stateU, &stateM }, { &gradient, &initStateU, &initStateM });
    BUILD_SINGLE_SELECTOR(gradient.dataType(), adaMaxUpdaterCudaLauncher, (blocksPerGrid, threadsPerBlock, context->getCudaStream(), 
                          gradient.specialBuffer(), gradient.specialShapeInfo(), initStateU.specialBuffer(),
                          initStateU.specialShapeInfo(), initStateM.specialBuffer(), initStateM.specialShapeInfo(),
                          update.specialBuffer(), update.specialShapeInfo(), stateU.specialBuffer(),
                          stateU.specialShapeInfo(), stateM.specialBuffer(), stateM.specialShapeInfo(),
                          dLr, dBeta1, dBeta2, dEpsilon, nIteration ), SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({ &update, &stateU, &stateM }, { &gradient, &initStateU, &initStateM });

    manager.synchronize();
}

}
}
}

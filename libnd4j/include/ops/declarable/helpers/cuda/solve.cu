/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
//  @author GS <sgazeos@gmail.com>
//
#include <system/op_boilerplate.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <execution/Threads.h>
#include <helpers/ConstantTadHelper.h>
#include "../triangular_solve.h"
#include "../lup.h"
#include "../solve.h"

namespace sd {
    namespace ops {
        namespace helpers {
    
            template <typename T>
            static SD_KERNEL void oneOnDiagonalKernel(T* ioBuf, sd::LongType const*  ioShape, sd::LongType const*  tadShape, sd::LongType const*  tadOffsets, sd::LongType batchNum, sd::LongType rowNum) {
                for (auto i = blockIdx.x; i < batchNum; i += gridDim.x) {
                    auto matrixPart = ioBuf + tadOffsets[i];
                    for (auto j = threadIdx.x; j < rowNum; j += blockDim.x) {
                        sd::LongType pos[] = {j, j};
                        auto offset = shape::getOffset(tadShape, pos);

                        matrixPart[offset] = T(1.f);
                    }
                }
            }

            template <typename T>
            static SD_KERNEL void restorePermutationsKernel(T* PBuf, sd::LongType const*  PShapeInfo, int const* permutationsBuf,
            sd::LongType const*  PTadShapeInfo, sd::LongType const*  PTadSOffsets, sd::LongType const*  permutationsTadShapeInfo, sd::LongType const*  permutationsTadOffsets, sd::LongType batchNum, sd::LongType rowNum) {
                for (auto batch = blockIdx.x; batch < batchNum; batch += gridDim.x) {
                    auto permutations = permutationsBuf + permutationsTadOffsets[batch];
                    auto P = PBuf + PTadSOffsets[batch];

                    for (auto row = threadIdx.x; row < rowNum; row += blockDim.x) {
                        //auto posX[] = {row};
                        sd::LongType posZ[] = {row, permutations[row]};
                        auto zOffset = shape::getOffset(PTadShapeInfo, posZ);
                        P[zOffset] = T(1.f);
                    }
                }
            }

            template <typename T>
            static sd::Status solveFunctor_(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput,
                                     bool adjoint, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {leftInput, rightInput});
                // stage 1: LU decomposition batched
                auto leftOutput = leftInput->ulike();
                auto permuShape = rightInput->getShapeAsVector(); permuShape.pop_back();
                auto permutations = NDArrayFactory::create<int>('c', permuShape, context);
                helpers::lu(context, leftInput, &leftOutput, &permutations);
                auto leftLower = leftOutput.dup();
                auto rightOutput = rightInput->ulike();
                auto leftLowerTad = ConstantTadHelper::getInstance().tadForDimensions(leftLower.shapeInfo(), {-2, -1});
                auto stream = context->getCudaStream();
                oneOnDiagonalKernel<T><<<128, 256, 256, *stream>>>(leftLower.dataBuffer()->specialAsT<T>(), leftLower.specialShapeInfo(), leftLowerTad.specialShapeInfo(), leftLowerTad.specialOffsets(), leftLowerTad.numberOfTads(), leftLower.sizeAt(-1));
                auto P = leftOutput.ulike(); P.nullify();
                auto PTad = ConstantTadHelper::getInstance().tadForDimensions(P.shapeInfo(), {-2, -1});
                auto permutationsTad = ConstantTadHelper::getInstance().tadForDimensions(permutations.shapeInfo(), {-1});
                restorePermutationsKernel<T><<<128, 256, 256, *stream>>>(P.dataBuffer()->specialAsT<T>(), P.specialShapeInfo(), permutations.dataBuffer()->specialAsT<int>(),
                        PTad.specialShapeInfo(), PTad.specialOffsets(), permutationsTad.specialShapeInfo(), permutationsTad.specialOffsets(), permutationsTad.numberOfTads(), permutations.sizeAt(-1));
                P.tickWriteDevice();
                auto rightPart = rightInput->ulike();
                MmulHelper::matmul(&P, rightInput, &rightPart, 0, 0);
                
                // stage 2: triangularSolveFunctor for Lower with given b
                helpers::triangularSolveFunctor(context, &leftLower, &rightPart, true, false, &rightOutput);
                // stage 3: triangularSolveFunctor for Upper with output of previous stage
                helpers::triangularSolveFunctor(context, &leftOutput, &rightOutput, false, false, output);
                NDArray::registerSpecialUse({output}, {leftInput, rightInput});

                return sd::Status::OK;
            }

            sd::Status solveFunctor(sd::LaunchContext * context, NDArray* leftInput, NDArray* rightInput, bool adjoint, NDArray* output) {
                BUILD_SINGLE_SELECTOR(leftInput->dataType(), return solveFunctor_, (context, leftInput, rightInput, adjoint, output), SD_FLOAT_TYPES);
            }

            template <typename T>
            static SD_KERNEL void adjointKernel(T* output, sd::LongType batchSize, sd::LongType rows, sd::LongType columns, sd::LongType const*  outputTads, 
                                                 sd::LongType const*  outputOffsets) {

                for (auto b = blockIdx.x; b < batchSize; b += gridDim.x) {
                    auto outputPart = output + outputOffsets[b];
                    for (auto r = threadIdx.x; r < rows; r += blockDim.x) {
                        for (auto c = threadIdx.y; c < r; c += blockDim.y) {
                            sd::LongType zPos[] = {r, c};
                            sd::LongType xPos[] = {c, r};
                            auto zIndex = shape::getOffset(outputTads, zPos);
                            auto xIndex = shape::getOffset(outputTads, xPos);
                            math::sd_swap(outputPart[zIndex], outputPart[xIndex]);
                        }
                    }
                }

            }

            template <typename T>
            static void adjointMatrix_(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
                NDArray::prepareSpecialUse({output}, {input});
                auto inputTads = ConstantTadHelper::getInstance().tadForDimensions(input->shapeInfo(), {-2, -1});
                auto outputTads = ConstantTadHelper::getInstance().tadForDimensions(output->shapeInfo(), {-2, -1});
                auto stream = context->getCudaStream();
                auto outputBuf = reinterpret_cast<T*>(output->specialBuffer());
                auto rows = input->sizeAt(-2);
                auto columns = input->sizeAt(-1);
                output->assign(input);
                adjointKernel<T><<<128, 256, 256, *stream>>>(outputBuf, outputTads.numberOfTads(), rows, columns, outputTads.specialShapeInfo(), outputTads.specialOffsets());
                NDArray::registerSpecialUse({output}, {input});
            }

            void adjointMatrix(sd::LaunchContext* context, NDArray const* input, NDArray* output) {
                BUILD_SINGLE_SELECTOR(input->dataType(), adjointMatrix_, (context, input, output), SD_FLOAT_TYPES);
            }

        }
    }
}

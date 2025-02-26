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
//  @author raver119@gmail.com
//   @author Adam Gibson
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_batched_gemm)

#include <ops/declarable/headers/blas.h>
#include <ops/declarable/helpers/batched_gemm.h>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(batched_gemm, -1, -1, false, 0, 9) {
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);


  if (transA == 0) transA = 111;

  if (transB == 0) transB = 111;

  if (transA == 1) transA = 112;

  if (transB == 1) transB = 112;


  REQUIRE_TRUE((transA == 111 || transA == 112) && (transB == 111 || transB == 112), 0,
               "BatchedGemm: valid values for transA and transB are: 0/1 or 111/112, for NoTrans/Trans respectively")
  REQUIRE_TRUE(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0, 0, "");


  auto alpha = INPUT_VARIABLE(0);
  auto beta = INPUT_VARIABLE(1);
  std::vector<NDArray*> vA(batchSize);
  std::vector<NDArray*> vB(batchSize);
  std::vector<NDArray*> vC(batchSize);

  auto firstType = INPUT_VARIABLE(0)->dataType();
  for (int e = 0; e < batchSize; e++) {
    vA[e] = INPUT_VARIABLE(e + 2);
    vB[e] = INPUT_VARIABLE(e + 2 + batchSize);
    vC[e] = OUTPUT_VARIABLE(e);

    REQUIRE_TRUE(firstType == vC[e]->dataType(), 0, "BatchedGemm: all inputs and outputs must have same data type");

    REQUIRE_TRUE(vA[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of A should be equal to 2", e);
    REQUIRE_TRUE(vB[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of B should be equal to 2", e);
    REQUIRE_TRUE(vC[e]->rankOf() == 2, 0, "BatchedGemm: batch %i, rank of C should be equal to 2", e);

    REQUIRE_TRUE(M == vA[e]->sizeAt(0), 0, "BatchedGemm: batch %i, number of A.rows() should be equal to M", e);
    REQUIRE_TRUE(N == vB[e]->sizeAt(1), 0, "BatchedGemm: batch %i, number of B.columns() should be equal to N", e);
    REQUIRE_TRUE(K == vA[e]->sizeAt(1) && K == vB[e]->sizeAt(0), 0,
                 "BatchedGemm: batch %i, number of A.columns() and B.rows() should be equal to K", e);
  }

  REQUIRE_TRUE(vA.size() == vB.size() && vA.size() == vC.size() && vA.size() == batchSize, 0,
               "BatchedGemm: mismatched numbers of A, B, C for unknown reason");

  sd::ops::helpers::bgemm(vA, vB, vC, alpha, beta, transA, transB, M, N, K, ldA, ldB, ldC);




  return sd::Status::OK;
};

DECLARE_SHAPE_FN(batched_gemm) {
  int transA = INT_ARG(0);
  int transB = INT_ARG(1);
  int M = INT_ARG(2);
  int N = INT_ARG(3);
  int K = INT_ARG(4);
  int ldA = INT_ARG(5);
  int ldB = INT_ARG(6);
  int ldC = INT_ARG(7);
  int batchSize = INT_ARG(8);

  auto firstType = ArrayOptions::dataType(inputShape->at(0));
  for (int e = 1; e < block.width(); e++) {
    REQUIRE_TRUE(firstType == ArrayOptions::dataType(inputShape->at(1)), 0,
                 "BatchedGemm: all inputs must have same data type");
  }

  auto shapeList = SHAPELIST();

  if (!(M > 0 && N > 0 && K > 0 && ldA > 0 && ldB > 0 && ldC > 0 && batchSize > 0)) {
    shapeList->push_back(
        ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShape->at(0)), 'c', {1, 1}));
    return shapeList;
  }

  std::vector<sd::LongType> shape({M, N});

  for (int e = 0; e < batchSize; e++) {
    auto newShape =
        ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inputShape->at(0)), 'f', shape);
    shapeList->push_back(newShape);
  }

  return shapeList;
}

DECLARE_TYPES(batched_gemm) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_FLOATS})
          //                    ->setAllowedInputTypes(1, {DataType::FLOAT32, DataType ::DOUBLE, DataType::HALF})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

}  // namespace ops
}  // namespace sd

#endif

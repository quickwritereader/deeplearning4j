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
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_cbow)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sg_cb.h>

namespace sd {
namespace ops {




CONFIGURABLE_OP_IMPL(cbow_inference, 6, 6, true, -2, -2) {
  //construct codes and indices from the IARGS
  //we do this to avoid serialization overhead from the JVM for frequently created small arrays
  auto numCodes = I_ARG(0);
  auto numIndices = I_ARG(1);
  auto numContext = I_ARG(2);
  auto numLockedWords = I_ARG(3);
  //2 for the codes, indices, context, locked words, 4 for the mandatory args such as target
  auto numMin = numIndices + numCodes + numCodes + numLockedWords + 4 + 4;
  std::vector<sd::LongType> *codes = new std::vector<sd::LongType>();
  std::vector<sd::LongType> *indices = new std::vector<sd::LongType>();
  std::vector<sd::LongType> *context = new std::vector<sd::LongType>();
  std::vector<sd::LongType> *lockedWords = new std::vector<sd::LongType>();

  int currIdx = 4;
  for(int i = 0; i < numCodes; i++) {
    codes->push_back(I_ARG(currIdx));
    currIdx++;
  }

  for(int i = 0; i < numIndices; i++) {
    indices->push_back(I_ARG(currIdx));
    currIdx++;
  }


  for(int i = 0; i < numContext; i++) {
    context->push_back(I_ARG(currIdx));
    currIdx++;
  }

  for(int i = 0; i < numLockedWords; i++) {
    lockedWords->push_back(I_ARG(currIdx));
    currIdx++;
  }



  const std::vector<sd::LongType> *indicesVec = indices;
  const std::vector<sd::LongType> *codesVec = codes;
  const std::vector<sd::LongType> *contextVec = context;
  const std::vector<sd::LongType> *lockedWordsVec = lockedWords;

  std::vector<sd::LongType> *indicesSize = new std::vector<sd::LongType>();
  indicesSize->push_back(indices->size());
  const std::vector<sd::LongType> *indicesShape = indicesSize;


  std::vector<sd::LongType> *codesSize = new std::vector<sd::LongType>();
  codesSize->push_back(codes->size());
  const std::vector<sd::LongType> *codesShape = codesSize;

  std::vector<sd::LongType> *contextSize = new std::vector<sd::LongType>();
  contextSize->push_back(contextSize->size());
  const std::vector<sd::LongType> *contextShape = contextSize;

  std::vector<sd::LongType> *lockedWordsSize = new std::vector<sd::LongType>();
  contextSize->push_back(lockedWords->size());
  const std::vector<sd::LongType> *lockedWordsShape = lockedWordsSize;


  auto indicesArrOne = NDArrayFactory::create('c',*indicesShape,*indicesVec);
  auto indicesArr = new NDArray(indicesArrOne);

  auto codesArrOne = NDArrayFactory::create('c',*codesShape,*codesVec);
  auto codesArr = new NDArray(codesArrOne);

  auto contextArrOne = NDArrayFactory::create('c',*contextShape,*contextVec);
  auto contextArr = new NDArray(contextArrOne);


  auto lockedWordsOne = NDArrayFactory::create('c',*lockedWordsShape,*lockedWordsVec);
  auto lockedWordsArr = new NDArray(lockedWordsOne);

  auto target = I_ARG(currIdx++);
  auto ngStarter = I_ARG(currIdx++);
  auto numLabels = I_ARG(currIdx++);
  auto randomValue = I_ARG(currIdx++);
  auto numWorkers = block.numI() > 0 ? INT_ARG(3) : omp_get_max_threads();
  auto nsRounds = block.numI() > 1 ? INT_ARG(4) : 0;

  auto alpha = T_ARG(0);




  auto syn0 = INPUT_VARIABLE(0);
  auto syn1 = INPUT_VARIABLE(1);
  auto syn1neg = INPUT_VARIABLE(2);

  auto expTable = INPUT_VARIABLE(3);
  auto negTable = INPUT_VARIABLE(4);

  auto inferenceVector = INPUT_VARIABLE(5);



  auto trainWords = block.numB() > 0 ? B_ARG(0) : true;
  auto isInference = block.numB() > 1 ? B_ARG(1) : false;

  REQUIRE_TRUE(block.isInplace(), 0, "CBOW: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "CBOW: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "CBOW: expTable must have the same data type as syn0 table");

  sd::ops::helpers::cbowInference(
                                 *syn0,
                                  *syn1,
                                *syn1neg,
                                   *expTable,
                                    *negTable,
                                    target,
                                  ngStarter,
                                   nsRounds,
                                 *contextArr,
                                *lockedWordsArr,
                              *indicesArr,
                              *codesArr,
                                 alpha,
                              randomValue,
                              numLabels,
                           *inferenceVector,
                              trainWords,
                                 numWorkers);

  return sd::Status::OK;
}

DECLARE_TYPES(cbow_inference) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}


CONFIGURABLE_OP_IMPL(cbow, 15, 15, true, 0, 0) {
  auto target = INPUT_VARIABLE(0);
  auto ngStarter = INPUT_VARIABLE(1);

  // required part
  auto context = INPUT_VARIABLE(2);
  auto indices = INPUT_VARIABLE(3);
  auto codes = INPUT_VARIABLE(4);

  auto syn0 = INPUT_VARIABLE(5);
  auto syn1 = INPUT_VARIABLE(6);
  auto syn1neg = INPUT_VARIABLE(7);

  auto expTable = INPUT_VARIABLE(8);
  auto negTable = INPUT_VARIABLE(9);

  auto alpha = INPUT_VARIABLE(10);
  auto randomValue = INPUT_VARIABLE(11);
  auto numLabels = INPUT_VARIABLE(12);

  auto lockedWords = INPUT_VARIABLE(13);

  auto inferenceVector = INPUT_VARIABLE(14);

  auto numWorkers = block.numI() > 0 ? INT_ARG(0) : omp_get_max_threads();
  auto nsRounds = block.numI() > 1 ? INT_ARG(1) : 0;

  auto trainWords = block.numB() > 0 ? B_ARG(0) : true;
  auto isInference = block.numB() > 1 ? B_ARG(1) : false;

  REQUIRE_TRUE(block.isInplace(), 0, "CBOW: this operation requires inplace execution only");

  REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0,
               "CBOW: all syn tables must have the same data type");
  REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0,
               "CBOW: expTable must have the same data type as syn0 table");

  sd::ops::helpers::cbow(*syn0, *syn1, *syn1neg, *expTable, *negTable, *target, *ngStarter, nsRounds, *context,
                         *lockedWords, *indices, *codes, *alpha, *randomValue, *numLabels, *inferenceVector, trainWords,
                         numWorkers);

  return sd::Status::OK;
}

DECLARE_TYPES(cbow) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, sd::DataType::INT32)
      ->setAllowedInputTypes(1, sd::DataType::INT32)
      ->setAllowedInputTypes(2, sd::DataType::INT32)
      ->setAllowedInputTypes(3, sd::DataType::INT32)
      ->setAllowedInputTypes(4, sd::DataType::INT8)
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedInputTypes(6, {ALL_FLOATS})
      ->setAllowedInputTypes(7, {ALL_FLOATS})
      ->setAllowedInputTypes(8, {ALL_FLOATS})
      ->setAllowedInputTypes(9, {ALL_FLOATS})
      ->setAllowedInputTypes(10, {ALL_FLOATS})
      ->setAllowedInputTypes(11, sd::DataType::INT64)
      ->setAllowedInputTypes(12, sd::DataType::INT32)
      ->setAllowedInputTypes(13, sd::DataType::INT32)
      ->setAllowedInputTypes(14, {ALL_FLOATS})
      ->setAllowedOutputTypes(sd::DataType::ANY);
}
}  // namespace ops
}  // namespace sd

#endif

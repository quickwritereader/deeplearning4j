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
// Created by raver119 on 02.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_expand_dims)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(expand_dims, 1, 1, false, 0, -2) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);
  sd::LongType axis = block.numI() > 0 ? INT_ARG(0) : INPUT_VARIABLE(1)->e<int>(0);

  if (axis < 0) axis += input->rankOf() + 1;

  REQUIRE_TRUE(axis >= 0 && axis <= input->rankOf(), 0,
               "ExpandDims: axis should be in range of 0...%i in this case, but got %i instead", input->rankOf() + 1,
               axis);


  if (input->ews() == 1 && output->ews() == 1 && input->ordering() == output->ordering()) {
    output->dataBuffer()->copyBufferFrom(*input->dataBuffer().get(),
                                         output->lengthOf() * DataTypeUtils::sizeOfElement(output->dataType()), 0,
                                         input->bufferOffset());
  } else {
    //the shape was already determined in the calculate shape info, just reshape to the same shape as the output
    auto tmp = input->reshape(input->ordering(), output->getShapeAsVector(),false);
    output->assign(tmp);
  }
  return sd::Status::OK;
}

DECLARE_TYPES(expand_dims) { getOpDescriptor()->setAllowedInputTypes(sd::DataType::ANY)->setSameMode(true); }

DECLARE_SHAPE_FN(expand_dims) {
  auto inShape = inputShape->at(0);

  // 0D scalar edge case
  if (shape::rank(inShape) == 0) {
    sd::LongType x = 1;
    auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), 'c', 1, &x);
    return SHAPELIST(newShape);
  }

  auto input = INPUT_VARIABLE(0);
  auto x_rank = shape::rank(inShape);
  char order = shape::order(inShape);

  sd::LongType axis = block.numI() > 0 ? INT_ARG(0) : INPUT_VARIABLE(1)->e<int>(0);
  REQUIRE_TRUE(axis >= 0 && axis <= input->rankOf(), 0,
               "ExpandDims: axis should be in range of 0...%i in this case, but got %i instead", input->rankOf() + 1,
               axis);

  if (axis < 0) axis += x_rank + 1;
  std::vector<sd::LongType> shape;
  for (int e = 0; e < x_rank; e++) shape.emplace_back(shape::shapeOf(inShape)[e]);

  shape.insert(shape.begin() + axis, 1);

  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(ArrayOptions::dataType(inShape), order, shape);
  return SHAPELIST(newShape);
}
}  // namespace ops
}  // namespace sd

#endif

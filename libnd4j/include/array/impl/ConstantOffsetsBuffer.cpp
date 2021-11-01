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
// @author raver119@gmail.com
//
#include <array/ConstantOffsetsBuffer.h>

namespace sd {
ConstantOffsetsBuffer::ConstantOffsetsBuffer(const std::shared_ptr<PointerWrapper> &primary) :
                       ConstantOffsetsBuffer(primary, std::shared_ptr<PointerWrapper>(nullptr)) {
  //
}

ConstantOffsetsBuffer::ConstantOffsetsBuffer(const std::shared_ptr<PointerWrapper> &primary,
                                             const std::shared_ptr<PointerWrapper> &special) {
  _primaryOffsets = primary;
  _specialOffsets = special;
}

const sd::LongType *ConstantOffsetsBuffer::primary() const {
  return reinterpret_cast<sd::LongType*>(_primaryOffsets->pointer());
}

const sd::LongType *ConstantOffsetsBuffer::special() const {
  return _specialOffsets ? reinterpret_cast<sd::LongType*>(_specialOffsets->pointer()) : nullptr;
}

const sd::LongType *ConstantOffsetsBuffer::platform() const {
#ifdef __CUDABLAS__
  return special();
#else
  return primary();
#endif // CUDABLAS
}

} // namespace sd

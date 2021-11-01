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
#include <array/ConstantDataBuffer.h>
#include <array/DataTypeUtils.h>

namespace sd {
ConstantDataBuffer::ConstantDataBuffer(
    const std::shared_ptr<PointerWrapper>& primary,
    uint64_t numEelements,
    DataType dtype) : ConstantDataBuffer(primary, {}, numEelements, dtype)   {
  //
}

ConstantDataBuffer::ConstantDataBuffer(
    const std::shared_ptr<PointerWrapper>& primary,
    const std::shared_ptr<PointerWrapper>& special,
    uint64_t numEelements,
    DataType dtype) : _primaryBuffer(primary), _specialBuffer(special), _length(numEelements) {
        _sizeOf = DataTypeUtils::sizeOf(dtype);
    }

    void* ConstantDataBuffer::primary() const {
        return _primaryBuffer->pointer();
    }

    void* ConstantDataBuffer::special() const {
        return _specialBuffer ? _specialBuffer->pointer() : nullptr;
    }

    uint8_t ConstantDataBuffer::sizeOf() const {
        return _sizeOf;
    }

    uint64_t ConstantDataBuffer::length() const {
        return _length;
    }

    ConstantDataBuffer::ConstantDataBuffer(const ConstantDataBuffer &other) {
        _primaryBuffer = other._primaryBuffer;
        _specialBuffer = other._specialBuffer;
        _length = other._length;
        _sizeOf = other._sizeOf;
    }

    template <typename T>
    T* ConstantDataBuffer::primaryAsT() const {
        return reinterpret_cast<T*>(_primaryBuffer->pointer());
    }
    template SD_LIB_EXPORT float* ConstantDataBuffer::primaryAsT<float>() const;
    template SD_LIB_EXPORT double* ConstantDataBuffer::primaryAsT<double>() const;
    template SD_LIB_EXPORT int* ConstantDataBuffer::primaryAsT<int>() const;
    template SD_LIB_EXPORT sd::LongType* ConstantDataBuffer::primaryAsT<sd::LongType>() const;

    template <typename T>
    T* ConstantDataBuffer::specialAsT() const {
        return reinterpret_cast<T*>(special());
    }
    template SD_LIB_EXPORT float* ConstantDataBuffer::specialAsT<float>() const;
    template SD_LIB_EXPORT double* ConstantDataBuffer::specialAsT<double>() const;
    template SD_LIB_EXPORT int* ConstantDataBuffer::specialAsT<int>() const;
    template SD_LIB_EXPORT sd::LongType* ConstantDataBuffer::specialAsT<sd::LongType>() const;

}

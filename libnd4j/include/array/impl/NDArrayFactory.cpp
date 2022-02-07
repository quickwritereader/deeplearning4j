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
// Created by GS <sgazeos@gmail.com> on 2018-12-20.
// @author Oleg Semeniv <oleg.semeniv@gmail.com>
//
#include <array/NDArrayFactory.h>
#include <exceptions/cuda_exception.h>
#include <graph/GraphExecutioner.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/LoopsCoordsHelper.h>
#include <helpers/ShapeUtils.h>
#include <helpers/StringUtils.h>
#include <legacy/NativeOps.h>

#include <type_traits>

namespace sd {

SD_LIB_EXPORT NDArray NDArrayFactory::create(const ShapeDescriptor& shapeDescriptor, sd::LaunchContext* context) {
  auto status = shapeDescriptor.validate();
  if (status != SHAPE_DESC_OK) {
    sd_printf("NDArrayFactory::create: ShapeDescriptor status code [%d]\n", status);
    throw std::invalid_argument("NDArrayFactory::create: invalid ShapeDescriptor ");
  }
  sd::LongType allocSize = shapeDescriptor.allocLength() * DataTypeUtils::sizeOfElement(shapeDescriptor.dataType());
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(allocSize, shapeDescriptor.dataType(), context->getWorkspace());
  NDArray result(buffer, shapeDescriptor, context);
  result.nullify();
  return result;
}

SD_LIB_EXPORT NDArray NDArrayFactory::create(const char order, const std::vector<sd::LongType>& shape,
                                             sd::DataType dataType, const std::vector<sd::LongType>& paddings,
                                             const std::vector<sd::LongType>& paddingOffsets,
                                             sd::LaunchContext* context) {
  int rank = shape.size();
  if (rank > SD_MAX_RANK) throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32");

  if (paddings.size() != rank) {
    throw std::invalid_argument("NDArrayFactory::create: paddings size should match rank ");
  }

  auto shapeDescriptor = ShapeDescriptor::paddedBufferDescriptor(dataType, order, shape, paddings);

  sd::LongType allocSize = shapeDescriptor.allocLength() * DataTypeUtils::sizeOfElement(shapeDescriptor.dataType());
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(allocSize, shapeDescriptor.dataType(), context->getWorkspace());

  // lets check offsets
  int check_size = paddingOffsets.size() < rank ? paddingOffsets.size() : rank;

  for (int i = 0; i < check_size; i++) {
    if (paddingOffsets[i] > paddings[i]) {
      throw std::invalid_argument(
          "NDArrayFactory::create: paddingOffsets numbers should not exceed corresponding paddings");
    }
  }

  sd::LongType offset = offset_from_coords(shapeDescriptor.stridesPtr(), paddingOffsets.data(), check_size);

  NDArray result(buffer, shapeDescriptor, context, offset);
  result.nullify();
  return result;
}

////////////////////////////////////////////////////////////////////////
template <>
SD_LIB_EXPORT NDArray NDArrayFactory::create<bool>(const char order, const std::vector<sd::LongType>& shape,
                                                   const std::vector<bool>& data, sd::LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

  ShapeDescriptor descriptor(sd::DataType::BOOL, order, shape);

  if (descriptor.arrLength() != data.size()) {
    sd_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(),
              descriptor.arrLength());
    throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
  }

  bool* hostBuffer = nullptr;
  ALLOCATE(hostBuffer, context->getWorkspace(), data.size(), bool);
  std::copy(data.begin(), data.end(), hostBuffer);

  std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(hostBuffer, data.size() * sizeof(bool),
                                                                    sd::DataType::BOOL, true, context->getWorkspace());

  NDArray result(buffer, descriptor, context);

  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const char order, const std::vector<sd::LongType>& shape, const std::vector<T>& data,
                               sd::LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32 !");

  ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shape);

  if (descriptor.arrLength() != data.size()) {
    sd_printf("NDArrayFactory::create: data size [%i] doesn't match shape length [%lld]\n", data.size(),
              descriptor.arrLength());
    throw std::runtime_error("NDArrayFactory::create: data size doesn't match shape");
  }

  std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(
      data.data(), DataTypeUtils::fromT<T>(), descriptor.arrLength() * sizeof(T), context->getWorkspace());

  NDArray result(buffer, descriptor, context);

  return result;
}

#define TMPL_INSTANTIATE_CREATE_A(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(const char order, const std::vector<sd::LongType>& shape, \
                                                      const std::vector<TYPE>& data, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_A(double)
TMPL_INSTANTIATE_CREATE_A(float)
TMPL_INSTANTIATE_CREATE_A(float16)
TMPL_INSTANTIATE_CREATE_A(bfloat16)
TMPL_INSTANTIATE_CREATE_A(sd::LongType)
TMPL_INSTANTIATE_CREATE_A(int)
TMPL_INSTANTIATE_CREATE_A(unsigned int)
TMPL_INSTANTIATE_CREATE_A(int8_t)
TMPL_INSTANTIATE_CREATE_A(int16_t)
TMPL_INSTANTIATE_CREATE_A(uint8_t)
TMPL_INSTANTIATE_CREATE_A(uint64_t)
TMPL_INSTANTIATE_CREATE_A(bool)

#undef TMPL_INSTANTIATE_CREATE_A
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<sd::LongType>& shape, sd::LaunchContext* context) {
  return create_(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray* NDArrayFactory::create_,
                      (const char order, const std::vector<sd::LongType>& shape, sd::LaunchContext* context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
template <typename T>
void NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<T>& vector) {
  memcpy(ptr, vector.data(), vector.size() * sizeof(T));
}

template <>
void SD_LIB_EXPORT NDArrayFactory::memcpyFromVector(void* ptr, const std::vector<bool>& vector) {
  auto p = reinterpret_cast<bool*>(ptr);
  for (sd::LongType e = 0; e < vector.size(); e++) p[e] = vector[e];
}


#define TMPL_INSTANTIATE_MEMCPY(TYPE) \
template SD_LIB_EXPORT void NDArrayFactory::memcpyFromVector<TYPE>(void* ptr, const std::vector<TYPE>& vector);

TMPL_INSTANTIATE_MEMCPY(double)
TMPL_INSTANTIATE_MEMCPY(float)
TMPL_INSTANTIATE_MEMCPY(float16)
TMPL_INSTANTIATE_MEMCPY(bfloat16)
TMPL_INSTANTIATE_MEMCPY(sd::LongType)
TMPL_INSTANTIATE_MEMCPY(int)
TMPL_INSTANTIATE_MEMCPY(int16_t)
TMPL_INSTANTIATE_MEMCPY(int8_t)
TMPL_INSTANTIATE_MEMCPY(uint8_t)
TMPL_INSTANTIATE_MEMCPY(bool)

#undef TMPL_INSTANTIATE_MEMCPY

#ifndef __JAVACPP_HACK__
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::valueOf(const std::initializer_list<sd::LongType>& shape, const T value, const char order,
                                 sd::LaunchContext* context) {
  return valueOf(std::vector<sd::LongType>(shape), value, order);
}

#define TMPL_INSTANTIATE_VALUEOF_A(TYPE) \
template SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf<TYPE>(const std::initializer_list<sd::LongType>& shape, \
                                                        const TYPE value, const char order, \
                                                        sd::LaunchContext* context);

TMPL_INSTANTIATE_VALUEOF_A(double)
TMPL_INSTANTIATE_VALUEOF_A(float)
TMPL_INSTANTIATE_VALUEOF_A(float16)
TMPL_INSTANTIATE_VALUEOF_A(bfloat16)
TMPL_INSTANTIATE_VALUEOF_A(sd::LongType)
TMPL_INSTANTIATE_VALUEOF_A(int)
TMPL_INSTANTIATE_VALUEOF_A(int16_t)
TMPL_INSTANTIATE_VALUEOF_A(int8_t)
TMPL_INSTANTIATE_VALUEOF_A(uint8_t)
TMPL_INSTANTIATE_VALUEOF_A(bool)

#undef TMPL_INSTANTIATE_VALUEOF_A

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const char order, const std::vector<sd::LongType>& shape,
                               const std::initializer_list<T>& data, sd::LaunchContext* context) {
  std::vector<T> vec(data);
  return create<T>(order, shape, vec, context);
}


#define TMPL_INSTANTIATE_CREATE_B(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(const char order, const std::vector<sd::LongType>& shape, \
                                                      const std::initializer_list<TYPE>& data, \
                                                      sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_B(double)
TMPL_INSTANTIATE_CREATE_B(float)
TMPL_INSTANTIATE_CREATE_B(float16)
TMPL_INSTANTIATE_CREATE_B(bfloat16)
TMPL_INSTANTIATE_CREATE_B(sd::LongType)
TMPL_INSTANTIATE_CREATE_B(int)
TMPL_INSTANTIATE_CREATE_B(unsigned int)
TMPL_INSTANTIATE_CREATE_B(int8_t)
TMPL_INSTANTIATE_CREATE_B(int16_t)
TMPL_INSTANTIATE_CREATE_B(uint8_t)
TMPL_INSTANTIATE_CREATE_B(uint16_t)
TMPL_INSTANTIATE_CREATE_B(uint64_t)
TMPL_INSTANTIATE_CREATE_B(bool)

#undef TMPL_INSTANTIATE_CREATE_B

#endif

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const T scalar, sd::LaunchContext* context) {
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(1 * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  NDArray* res = new NDArray(buffer, ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()), context);

  res->bufferAsT<T>()[0] = scalar;

  res->tickWriteHost();
  res->syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_C(TYPE) \
template SD_LIB_EXPORT NDArray* NDArrayFactory::create_<TYPE>(const TYPE scalar, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_C(double)
TMPL_INSTANTIATE_CREATE_C(float)
TMPL_INSTANTIATE_CREATE_C(float16)
TMPL_INSTANTIATE_CREATE_C(bfloat16)
TMPL_INSTANTIATE_CREATE_C(sd::LongType)
TMPL_INSTANTIATE_CREATE_C(int)
TMPL_INSTANTIATE_CREATE_C(unsigned int)
TMPL_INSTANTIATE_CREATE_C(int8_t)
TMPL_INSTANTIATE_CREATE_C(int16_t)
TMPL_INSTANTIATE_CREATE_C(uint8_t)
TMPL_INSTANTIATE_CREATE_C(uint16_t)
TMPL_INSTANTIATE_CREATE_C(uint64_t)
TMPL_INSTANTIATE_CREATE_C(bool)

#undef TMPL_INSTANTIATE_CREATE_C

template <typename T>
NDArray NDArrayFactory::create(sd::DataType type, const T scalar, sd::LaunchContext* context) {
  if (type == DataTypeUtils::fromT<T>()) return NDArrayFactory::create(scalar, context);

  NDArray res(type, context);
  res.p(0, scalar);
  res.syncToDevice();

  return res;
}
//    BUILD_DOUBLE_TEMPLATE(template SD_LIB_EXPORT NDArray NDArrayFactory::create, (DataType type, const T scalar,
//    sd::LaunchContext * context), SD_COMMON_TYPES_ALL);

#define TMPL_INSTANTIATE_CREATE_D(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(DataType type, const TYPE scalar, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_D(double)
TMPL_INSTANTIATE_CREATE_D(float)
TMPL_INSTANTIATE_CREATE_D(float16)
TMPL_INSTANTIATE_CREATE_D(bfloat16)
TMPL_INSTANTIATE_CREATE_D(sd::LongType)
TMPL_INSTANTIATE_CREATE_D(int)
TMPL_INSTANTIATE_CREATE_D(unsigned int)
TMPL_INSTANTIATE_CREATE_D(int8_t)
TMPL_INSTANTIATE_CREATE_D(int16_t)
TMPL_INSTANTIATE_CREATE_D(uint8_t)
TMPL_INSTANTIATE_CREATE_D(uint16_t)
TMPL_INSTANTIATE_CREATE_D(uint64_t)
TMPL_INSTANTIATE_CREATE_D(bool)

#undef TMPL_INSTANTIATE_CREATE_D

template <typename T>
NDArray NDArrayFactory::create(const T scalar, sd::LaunchContext* context) {
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(1 * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  NDArray res(buffer, ShapeDescriptor::scalarDescriptor(DataTypeUtils::fromT<T>()), context);

  res.bufferAsT<T>()[0] = scalar;

  res.tickWriteHost();
  res.syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_E(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(const TYPE scalar, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_E(double)
TMPL_INSTANTIATE_CREATE_E(float)
TMPL_INSTANTIATE_CREATE_E(float16)
TMPL_INSTANTIATE_CREATE_E(bfloat16)
TMPL_INSTANTIATE_CREATE_E(sd::LongType)
TMPL_INSTANTIATE_CREATE_E(int)
TMPL_INSTANTIATE_CREATE_E(unsigned int)
TMPL_INSTANTIATE_CREATE_E(int8_t)
TMPL_INSTANTIATE_CREATE_E(int16_t)
TMPL_INSTANTIATE_CREATE_E(uint8_t)
TMPL_INSTANTIATE_CREATE_E(uint16_t)
TMPL_INSTANTIATE_CREATE_E(uint64_t)
TMPL_INSTANTIATE_CREATE_E(bool)

#undef TMPL_INSTANTIATE_CREATE_E

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::create_(const char order, const std::vector<sd::LongType>& shape, const std::vector<T>& data,
                                 sd::LaunchContext* context) {
  return new NDArray(NDArrayFactory::create<T>(order, shape, data, context));
}

#define TMPL_INSTANTIATE_CREATE_F(TYPE) \
template SD_LIB_EXPORT NDArray* NDArrayFactory::create_<TYPE>(const char order, const std::vector<sd::LongType>& shape, \
                                                        const std::vector<TYPE>& data, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_F(double)
TMPL_INSTANTIATE_CREATE_F(float)
TMPL_INSTANTIATE_CREATE_F(float16)
TMPL_INSTANTIATE_CREATE_F(bfloat16)
TMPL_INSTANTIATE_CREATE_F(sd::LongType)
TMPL_INSTANTIATE_CREATE_F(int)
TMPL_INSTANTIATE_CREATE_F(unsigned int)
TMPL_INSTANTIATE_CREATE_F(int8_t)
TMPL_INSTANTIATE_CREATE_F(int16_t)
TMPL_INSTANTIATE_CREATE_F(uint8_t)
TMPL_INSTANTIATE_CREATE_F(uint16_t)
TMPL_INSTANTIATE_CREATE_F(uint64_t)
TMPL_INSTANTIATE_CREATE_F(bool)

#undef TMPL_INSTANTIATE_CREATE_F

////////////////////////////////////////////////////////////////////////
template <>
SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf(const std::vector<sd::LongType>& shape, NDArray* value, const char order,
                                               sd::LaunchContext* context) {
  auto result = create_(order, shape, value->dataType(), context);
  result->assign(*value);
  return result;
}

template <>
SD_LIB_EXPORT NDArray* NDArrayFactory::valueOf(const std::vector<sd::LongType>& shape, NDArray& value, const char order,
                                               sd::LaunchContext* context) {
  auto result = create_(order, shape, value.dataType(), context);
  result->assign(value);
  return result;
}

template <typename T>
NDArray* NDArrayFactory::valueOf(const std::vector<sd::LongType>& shape, const T value, const char order,
                                 sd::LaunchContext* context) {
  auto result = create_(order, shape, DataTypeUtils::fromT<T>());
  result->assign(value);
  return result;
}

#define TMPL_INSTANTIATE_VALUEOF(TYPE) \
template SD_LIB_EXPORT NDArray* \
NDArrayFactory::valueOf<TYPE>(const std::vector<sd::LongType>& shape, const TYPE value, \
                                                        const char order, sd::LaunchContext* context);

TMPL_INSTANTIATE_VALUEOF(double)
TMPL_INSTANTIATE_VALUEOF(float)
TMPL_INSTANTIATE_VALUEOF(float16)
TMPL_INSTANTIATE_VALUEOF(bfloat16)
TMPL_INSTANTIATE_VALUEOF(sd::LongType)
TMPL_INSTANTIATE_VALUEOF(int)
TMPL_INSTANTIATE_VALUEOF(int16_t)
TMPL_INSTANTIATE_VALUEOF(int8_t)
TMPL_INSTANTIATE_VALUEOF(uint8_t)
TMPL_INSTANTIATE_VALUEOF(bool)

#undef TMPL_INSTANTIATE_VALUEOF

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::linspace(const T from, const T to, const sd::LongType numElements) {
  NDArray* result = NDArrayFactory::vector<T>(numElements);
  // TO DO: linspace should be executed on DEVICE, but only CPU version implemnted!
  for (sd::LongType e = 0; e < numElements; e++) {
    T step = (T)e / ((T)numElements - (T)1);
    result->p<T>(e, (from * ((T)1 - step) + step * to));
  }
  result->syncToDevice();

  return result;
}

#define TMPL_INSTANTIATE_LINSPACE(TYPE) \
template SD_LIB_EXPORT NDArray* NDArrayFactory::linspace<TYPE>(const TYPE from, const TYPE to, \
                                                         const sd::LongType numElements);


TMPL_INSTANTIATE_LINSPACE(double)
TMPL_INSTANTIATE_LINSPACE(float)
TMPL_INSTANTIATE_LINSPACE(float16)
TMPL_INSTANTIATE_LINSPACE(bfloat16)
TMPL_INSTANTIATE_LINSPACE(sd::LongType)
TMPL_INSTANTIATE_LINSPACE(int)
TMPL_INSTANTIATE_LINSPACE(int16_t)
TMPL_INSTANTIATE_LINSPACE(int8_t)
TMPL_INSTANTIATE_LINSPACE(uint8_t)
TMPL_INSTANTIATE_LINSPACE(bool)

#undef TMPL_INSTANTIATE_LINSPACE
////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::vector(sd::LongType length, const T value, sd::LaunchContext* context) {
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(length * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  auto res = new NDArray(buffer, ShapeDescriptor::vectorDescriptor(length, DataTypeUtils::fromT<T>()), context);

  if (value == (T)0.0f)
    res->nullify();
  else
    res->assign(value);

  return res;
}

#define TMPL_INSTANTIATE_VECTOR(TYPE) \
template SD_LIB_EXPORT NDArray* NDArrayFactory::vector<TYPE>(sd::LongType length, const TYPE startingValue, \
                                                       sd::LaunchContext* context);

TMPL_INSTANTIATE_VECTOR(double)
TMPL_INSTANTIATE_VECTOR(float)
TMPL_INSTANTIATE_VECTOR(float16)
TMPL_INSTANTIATE_VECTOR(bfloat16)
TMPL_INSTANTIATE_VECTOR(sd::LongType)
TMPL_INSTANTIATE_VECTOR(int)
TMPL_INSTANTIATE_VECTOR(int16_t)
TMPL_INSTANTIATE_VECTOR(int8_t)
TMPL_INSTANTIATE_VECTOR(uint8_t)
TMPL_INSTANTIATE_VECTOR(bool)

#undef TMPL_INSTANTIATE_VECTOR

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const char order, const std::initializer_list<sd::LongType>& shape,
                               sd::LaunchContext* context) {
  std::vector<sd::LongType> vec(shape);
  return create<T>(order, vec, context);
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray NDArrayFactory::create,
                      (const char, const std::initializer_list<sd::LongType>&, sd::LaunchContext* context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const char order, const std::vector<sd::LongType>& shape, sd::LaunchContext* context) {
  return create(order, shape, DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray NDArrayFactory::create,
                      (const char order, const std::vector<sd::LongType>& shape, sd::LaunchContext* context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(const char order, const std::vector<sd::LongType>& shape, sd::DataType dtype,
                               sd::LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    throw std::invalid_argument("NDArrayFactory::create: rank of NDArray can't exceed 32");

  ShapeDescriptor descriptor(dtype, order, shape);

  std::shared_ptr<DataBuffer> buffer = std::make_shared<DataBuffer>(
      descriptor.arrLength() * DataTypeUtils::sizeOfElement(dtype), dtype, context->getWorkspace());

  NDArray result(buffer, descriptor, context);

  result.nullify();

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::create(sd::DataType dtype, sd::LaunchContext* context) {
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(DataTypeUtils::sizeOfElement(dtype), dtype, context->getWorkspace(), true);

  NDArray res(buffer, ShapeDescriptor::scalarDescriptor(dtype), context);

  res.nullify();

  return res;
}

NDArray* NDArrayFactory::create_(sd::DataType dtype, sd::LaunchContext* context) {
  auto result = new NDArray();
  *result = NDArrayFactory::create(dtype, context);
  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(const std::vector<T>& values, sd::LaunchContext* context) {
  std::shared_ptr<DataBuffer> buffer =
      std::make_shared<DataBuffer>(values.size() * sizeof(T), DataTypeUtils::fromT<T>(), context->getWorkspace(), true);

  NDArray res(buffer, ShapeDescriptor::vectorDescriptor(values.size(), DataTypeUtils::fromT<T>()), context);

  memcpyFromVector<T>(res.buffer(), values);

  res.tickWriteHost();
  res.syncToDevice();

  return res;
}

#define TMPL_INSTANTIATE_CREATE_G(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(const std::vector<TYPE>& values, sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_G(double)
TMPL_INSTANTIATE_CREATE_G(float)
TMPL_INSTANTIATE_CREATE_G(float16)
TMPL_INSTANTIATE_CREATE_G(bfloat16)
TMPL_INSTANTIATE_CREATE_G(sd::LongType)
TMPL_INSTANTIATE_CREATE_G(int)
TMPL_INSTANTIATE_CREATE_G(unsigned int)
TMPL_INSTANTIATE_CREATE_G(int8_t)
TMPL_INSTANTIATE_CREATE_G(int16_t)
TMPL_INSTANTIATE_CREATE_G(uint8_t)
TMPL_INSTANTIATE_CREATE_G(uint16_t)
TMPL_INSTANTIATE_CREATE_G(uint64_t)
TMPL_INSTANTIATE_CREATE_G(bool)

#undef TMPL_INSTANTIATE_CREATE_G

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray* NDArrayFactory::empty_(sd::LaunchContext* context) {
  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(DataTypeUtils::fromT<T>(), context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  auto result = new NDArray(nullptr, shapeInfo, context, false);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray* NDArrayFactory::empty_, (sd::LaunchContext * context),
                      SD_COMMON_TYPES_ALL);

NDArray* NDArrayFactory::empty_(sd::DataType dataType, sd::LaunchContext* context) {
  if (context == nullptr) context = sd::LaunchContext ::defaultContext();

  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  auto result = new NDArray(nullptr, shapeInfo, context, false);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::empty(sd::LaunchContext* context) {
  return empty(DataTypeUtils::fromT<T>(), context);
}
BUILD_SINGLE_TEMPLATE(template SD_LIB_EXPORT NDArray NDArrayFactory::empty, (sd::LaunchContext * context),
                      SD_COMMON_TYPES_ALL);

////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::empty(sd::DataType dataType, sd::LaunchContext* context) {
  auto shapeInfo = ShapeBuilders::createScalarShapeInfo(dataType, context->getWorkspace());
  ArrayOptions::setPropertyBit(shapeInfo, ARRAY_EMPTY);
  NDArray result(nullptr, shapeInfo, context, false);

  RELEASE(shapeInfo, context->getWorkspace());

  return result;
}

////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::valueOf(const std::vector<sd::LongType>& shape, const NDArray& value, const char order,
                                 sd::LaunchContext* context) {
  auto res = NDArrayFactory::create_(order, shape, value.dataType(), context);
  res->assign(const_cast<NDArray&>(value));
  return res;
}

////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::create_(const char order, const std::vector<sd::LongType>& shape, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return new NDArray(order, shape, dataType, context);
}

////////////////////////////////////////////////////////////////////////
template <typename T>
NDArray NDArrayFactory::create(T* buffer, const char order, const std::initializer_list<sd::LongType>& shape,
                               sd::LaunchContext* context) {
  if ((int)shape.size() > SD_MAX_RANK)
    throw std::invalid_argument("NDArrayFactory::create: Rank of NDArray can't exceed 32");

  std::vector<sd::LongType> shp(shape);
  ShapeDescriptor descriptor(DataTypeUtils::fromT<T>(), order, shp);

  std::shared_ptr<DataBuffer> pBuffer = std::make_shared<DataBuffer>(
      buffer, descriptor.arrLength() * sizeof(T), descriptor.dataType(), false, context->getWorkspace());

  NDArray result(pBuffer, descriptor, context);

  return result;
}

#define TMPL_INSTANTIATE_CREATE_H(TYPE) \
template SD_LIB_EXPORT NDArray NDArrayFactory::create<TYPE>(TYPE* buffer, const char order, \
                                                      const std::initializer_list<sd::LongType>& shape,  \
                                                      sd::LaunchContext* context);

TMPL_INSTANTIATE_CREATE_H(double)
TMPL_INSTANTIATE_CREATE_H(float)
TMPL_INSTANTIATE_CREATE_H(float16)
TMPL_INSTANTIATE_CREATE_H(bfloat16)
TMPL_INSTANTIATE_CREATE_H(sd::LongType)
TMPL_INSTANTIATE_CREATE_H(int)
TMPL_INSTANTIATE_CREATE_H(unsigned int)
TMPL_INSTANTIATE_CREATE_H(int8_t)
TMPL_INSTANTIATE_CREATE_H(int16_t)
TMPL_INSTANTIATE_CREATE_H(uint8_t)
TMPL_INSTANTIATE_CREATE_H(uint16_t)
TMPL_INSTANTIATE_CREATE_H(uint64_t)
TMPL_INSTANTIATE_CREATE_H(bool)

#undef TMPL_INSTANTIATE_CREATE_H


/////////////////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const char16_t* u16string, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(u16string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char16_t* u16string, sd::DataType dtype, sd::LaunchContext* context) {
  return string_(std::u16string(u16string), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::u16string& u16string, sd::DataType dtype, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(u16string, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::u16string& u16string, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(u16string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const char32_t* u32string, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(u32string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char32_t* u32string, sd::DataType dtype, sd::LaunchContext* context) {
  return string_(std::u32string(u32string), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::u32string& u32string, sd::DataType dtype, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(u32string, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::u32string& u32string, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(u32string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const char* str, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(str, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const char* str, sd::DataType dtype, sd::LaunchContext* context) {
  return string_(std::string(str), dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::string& str, sd::DataType dtype, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(str, dtype, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::string& str, sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(str, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape,
                               const std::initializer_list<const char*>& strings, sd::DataType dataType,
                               sd::LaunchContext* context) {
  return NDArray(shape, std::vector<const char*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<const char*>& strings,
                               sd::DataType dataType, sd::LaunchContext* context) {
  return NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::initializer_list<std::string>& string,
                               sd::DataType dataType, sd::LaunchContext* context) {
  return NDArray(shape, std::vector<std::string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<const char*>& strings, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<const char*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<const char*>& strings,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  std::vector<std::string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::string(s);

  return NDArrayFactory::string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<std::string>& string, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<std::string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<std::string>& string,
                               sd::DataType dataType, sd::LaunchContext* context) {
  return NDArray(shape, string, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<std::string>& string,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(shape, string, dataType, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape,
                               const std::initializer_list<const char16_t*>& strings, sd::DataType dataType,
                               sd::LaunchContext* context) {
  return NDArray(shape, std::vector<const char16_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<const char16_t*>& strings,
                               sd::DataType dataType, sd::LaunchContext* context) {
  return NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape,
                               const std::initializer_list<std::u16string>& string, sd::DataType dataType,
                               sd::LaunchContext* context) {
  return NDArray(shape, std::vector<std::u16string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<const char16_t*>& strings, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<const char16_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<const char16_t*>& strings,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  std::vector<std::u16string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::u16string(s);

  return NDArrayFactory::string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<std::u16string>& string, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<std::u16string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<std::u16string>& string,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(shape, string, dataType, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<std::u16string>& string,
                               sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(shape, string, dtype, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape,
                               const std::initializer_list<const char32_t*>& strings, sd::DataType dataType,
                               sd::LaunchContext* context) {
  return NDArray(shape, std::vector<const char32_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<const char32_t*>& strings,
                               sd::DataType dataType, sd::LaunchContext* context) {
  return NDArray(shape, strings, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape,
                               const std::initializer_list<std::u32string>& string, sd::DataType dataType,
                               sd::LaunchContext* context) {
  return NDArray(shape, std::vector<std::u32string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<const char32_t*>& strings, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<const char32_t*>(strings), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<const char32_t*>& strings,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  std::vector<std::u32string> vec(strings.size());
  int cnt = 0;
  for (auto s : strings) vec[cnt++] = std::u32string(s);
  return NDArrayFactory::string_(shape, vec, dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape,
                                 const std::initializer_list<std::u32string>& string, sd::DataType dataType,
                                 sd::LaunchContext* context) {
  return NDArrayFactory::string_(shape, std::vector<std::u32string>(string), dataType, context);
}
/////////////////////////////////////////////////////////////////////////
NDArray* NDArrayFactory::string_(const std::vector<sd::LongType>& shape, const std::vector<std::u32string>& string,
                                 sd::DataType dataType, sd::LaunchContext* context) {
  auto res = new NDArray();
  *res = NDArray(shape, string, dataType, context);
  return res;
}
/////////////////////////////////////////////////////////////////////////
NDArray NDArrayFactory::string(const std::vector<sd::LongType>& shape, const std::vector<std::u32string>& string,
                               sd::DataType dtype, sd::LaunchContext* context) {
  return NDArray(shape, string, dtype, context);
}

NDArray NDArrayFactory::fromNpyFile(const char* fileName) {
  auto size = sd::graph::getFileSize(fileName);
  if (size < 0) throw std::runtime_error("File doesn't exit");

  auto pNPY = reinterpret_cast<char*>(::numpyFromFile(std::string(fileName)));

  auto nBuffer = reinterpret_cast<void*>(::dataPointForNumpy(pNPY));
  auto shape = reinterpret_cast<sd::LongType*>(::shapeBufferForNumpy(pNPY));

  auto length = shape::length(shape);
  int8_t* buffer = nullptr;
  sd::memory::Workspace* workspace = nullptr;
  auto byteLen = length * DataTypeUtils::sizeOfElement(ArrayOptions::dataType(shape));

  ALLOCATE(buffer, workspace, byteLen, int8_t);
  memcpy(buffer, nBuffer, byteLen);

  free(pNPY);

  return NDArray(buffer, shape, LaunchContext::defaultContext(), true);
}
}  // namespace sd

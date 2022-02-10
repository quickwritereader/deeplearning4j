/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


#ifndef FLATBUFFERS_GENERATED_ARRAY_SD_GRAPH_H_
#define FLATBUFFERS_GENERATED_ARRAY_SD_GRAPH_H_

#include "flatbuffers/flatbuffers.h"

namespace sd {
namespace graph {

struct FlatArray;
struct FlatArrayBuilder;

enum ByteOrder {
  ByteOrder_LE = 0,
  ByteOrder_BE = 1,
  ByteOrder_MIN = ByteOrder_LE,
  ByteOrder_MAX = ByteOrder_BE
};

inline const ByteOrder (&EnumValuesByteOrder())[2] {
  static const ByteOrder values[] = {
    ByteOrder_LE,
    ByteOrder_BE
  };
  return values;
}

inline const char * const *EnumNamesByteOrder() {
  static const char * const names[3] = {
    "LE",
    "BE",
    nullptr
  };
  return names;
}

inline const char *EnumNameByteOrder(ByteOrder e) {
  if (flatbuffers::IsOutRange(e, ByteOrder_LE, ByteOrder_BE)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesByteOrder()[index];
}

enum DType {
  DType_INHERIT = 0,
  DType_BOOL = 1,
  DType_FLOAT8 = 2,
  DType_HALF = 3,
  DType_HALF2 = 4,
  DType_FLOAT = 5,
  DType_DOUBLE = 6,
  DType_INT8 = 7,
  DType_INT16 = 8,
  DType_INT32 = 9,
  DType_INT64 = 10,
  DType_UINT8 = 11,
  DType_UINT16 = 12,
  DType_UINT32 = 13,
  DType_UINT64 = 14,
  DType_QINT8 = 15,
  DType_QINT16 = 16,
  DType_BFLOAT16 = 17,
  DType_UTF8 = 50,
  DType_UTF16 = 51,
  DType_UTF32 = 52,
  DType_MIN = DType_INHERIT,
  DType_MAX = DType_UTF32
};

inline const DType (&EnumValuesDType())[21] {
  static const DType values[] = {
    DType_INHERIT,
    DType_BOOL,
    DType_FLOAT8,
    DType_HALF,
    DType_HALF2,
    DType_FLOAT,
    DType_DOUBLE,
    DType_INT8,
    DType_INT16,
    DType_INT32,
    DType_INT64,
    DType_UINT8,
    DType_UINT16,
    DType_UINT32,
    DType_UINT64,
    DType_QINT8,
    DType_QINT16,
    DType_BFLOAT16,
    DType_UTF8,
    DType_UTF16,
    DType_UTF32
  };
  return values;
}

inline const char * const *EnumNamesDType() {
  static const char * const names[54] = {
    "INHERIT",
    "BOOL",
    "FLOAT8",
    "HALF",
    "HALF2",
    "FLOAT",
    "DOUBLE",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "QINT8",
    "QINT16",
    "BFLOAT16",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "UTF8",
    "UTF16",
    "UTF32",
    nullptr
  };
  return names;
}

inline const char *EnumNameDType(DType e) {
  if (flatbuffers::IsOutRange(e, DType_INHERIT, DType_UTF32)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesDType()[index];
}

enum LossReduce {
  LossReduce_NONE = 0,
  LossReduce_SUM = 1,
  LossReduce_MEAN_BY_WEIGHT = 2,
  LossReduce_MEAN_BY_NONZERO_WEIGHT_COUNT = 3,
  LossReduce_MIN = LossReduce_NONE,
  LossReduce_MAX = LossReduce_MEAN_BY_NONZERO_WEIGHT_COUNT
};

inline const LossReduce (&EnumValuesLossReduce())[4] {
  static const LossReduce values[] = {
    LossReduce_NONE,
    LossReduce_SUM,
    LossReduce_MEAN_BY_WEIGHT,
    LossReduce_MEAN_BY_NONZERO_WEIGHT_COUNT
  };
  return values;
}

inline const char * const *EnumNamesLossReduce() {
  static const char * const names[5] = {
    "NONE",
    "SUM",
    "MEAN_BY_WEIGHT",
    "MEAN_BY_NONZERO_WEIGHT_COUNT",
    nullptr
  };
  return names;
}

inline const char *EnumNameLossReduce(LossReduce e) {
  if (flatbuffers::IsOutRange(e, LossReduce_NONE, LossReduce_MEAN_BY_NONZERO_WEIGHT_COUNT)) return "";
  const size_t index = static_cast<size_t>(e);
  return EnumNamesLossReduce()[index];
}

struct FlatArray FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  typedef FlatArrayBuilder Builder;
  enum FlatBuffersVTableOffset FLATBUFFERS_VTABLE_UNDERLYING_TYPE {
    VT_SHAPE = 4,
    VT_BUFFER = 6,
    VT_DTYPE = 8,
    VT_BYTEORDER = 10
  };
  const flatbuffers::Vector<int64_t> *shape() const {
    return GetPointer<const flatbuffers::Vector<int64_t> *>(VT_SHAPE);
  }
  const flatbuffers::Vector<int8_t> *buffer() const {
    return GetPointer<const flatbuffers::Vector<int8_t> *>(VT_BUFFER);
  }
  sd::graph::DType dtype() const {
    return static_cast<sd::graph::DType>(GetField<int8_t>(VT_DTYPE, 0));
  }
  sd::graph::ByteOrder byteOrder() const {
    return static_cast<sd::graph::ByteOrder>(GetField<int8_t>(VT_BYTEORDER, 0));
  }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyOffset(verifier, VT_SHAPE) &&
           verifier.VerifyVector(shape()) &&
           VerifyOffset(verifier, VT_BUFFER) &&
           verifier.VerifyVector(buffer()) &&
           VerifyField<int8_t>(verifier, VT_DTYPE) &&
           VerifyField<int8_t>(verifier, VT_BYTEORDER) &&
           verifier.EndTable();
  }
};

struct FlatArrayBuilder {
  typedef FlatArray Table;
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_shape(flatbuffers::Offset<flatbuffers::Vector<int64_t>> shape) {
    fbb_.AddOffset(FlatArray::VT_SHAPE, shape);
  }
  void add_buffer(flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer) {
    fbb_.AddOffset(FlatArray::VT_BUFFER, buffer);
  }
  void add_dtype(sd::graph::DType dtype) {
    fbb_.AddElement<int8_t>(FlatArray::VT_DTYPE, static_cast<int8_t>(dtype), 0);
  }
  void add_byteOrder(sd::graph::ByteOrder byteOrder) {
    fbb_.AddElement<int8_t>(FlatArray::VT_BYTEORDER, static_cast<int8_t>(byteOrder), 0);
  }
  explicit FlatArrayBuilder(flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  FlatArrayBuilder &operator=(const FlatArrayBuilder &);
  flatbuffers::Offset<FlatArray> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = flatbuffers::Offset<FlatArray>(end);
    return o;
  }
};

inline flatbuffers::Offset<FlatArray> CreateFlatArray(
    flatbuffers::FlatBufferBuilder &_fbb,
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> shape = 0,
    flatbuffers::Offset<flatbuffers::Vector<int8_t>> buffer = 0,
    sd::graph::DType dtype = sd::graph::DType_INHERIT,
    sd::graph::ByteOrder byteOrder = sd::graph::ByteOrder_LE) {
  FlatArrayBuilder builder_(_fbb);
  builder_.add_buffer(buffer);
  builder_.add_shape(shape);
  builder_.add_byteOrder(byteOrder);
  builder_.add_dtype(dtype);
  return builder_.Finish();
}

inline flatbuffers::Offset<FlatArray> CreateFlatArrayDirect(
    flatbuffers::FlatBufferBuilder &_fbb,
    const std::vector<int64_t> *shape = nullptr,
    const std::vector<int8_t> *buffer = nullptr,
    sd::graph::DType dtype = sd::graph::DType_INHERIT,
    sd::graph::ByteOrder byteOrder = sd::graph::ByteOrder_LE) {
  auto shape__ = shape ? _fbb.CreateVector<int64_t>(*shape) : 0;
  auto buffer__ = buffer ? _fbb.CreateVector<int8_t>(*buffer) : 0;
  return sd::graph::CreateFlatArray(
      _fbb,
      shape__,
      buffer__,
      dtype,
      byteOrder);
}

inline const sd::graph::FlatArray *GetFlatArray(const void *buf) {
  return flatbuffers::GetRoot<sd::graph::FlatArray>(buf);
}

inline const sd::graph::FlatArray *GetSizePrefixedFlatArray(const void *buf) {
  return flatbuffers::GetSizePrefixedRoot<sd::graph::FlatArray>(buf);
}

inline bool VerifyFlatArrayBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifyBuffer<sd::graph::FlatArray>(nullptr);
}

inline bool VerifySizePrefixedFlatArrayBuffer(
    flatbuffers::Verifier &verifier) {
  return verifier.VerifySizePrefixedBuffer<sd::graph::FlatArray>(nullptr);
}

inline void FinishFlatArrayBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatArray> root) {
  fbb.Finish(root);
}

inline void FinishSizePrefixedFlatArrayBuffer(
    flatbuffers::FlatBufferBuilder &fbb,
    flatbuffers::Offset<sd::graph::FlatArray> root) {
  fbb.FinishSizePrefixed(root);
}

}  // namespace graph
}  // namespace sd

#endif  // FLATBUFFERS_GENERATED_ARRAY_SD_GRAPH_H_

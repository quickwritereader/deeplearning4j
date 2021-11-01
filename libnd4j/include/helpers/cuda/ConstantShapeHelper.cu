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
#include "../ConstantShapeHelper.h"
#include <exceptions/cuda_exception.h>
#include <array/ShapeDescriptor.h>
#include <helpers/ShapeBuilders.h>
#include <execution/AffinityManager.h>
#include <helpers/ConstantHelper.h>
#include <helpers/ShapeUtils.h>
#include <array/PrimaryPointerDeallocator.h>
#include <array/CudaPointerDeallocator.h>

namespace sd {

    ConstantShapeHelper::ConstantShapeHelper() {
        auto numDevices = AffinityManager::numberOfDevices();

        _cache.resize(numDevices);
        for (int e = 0; e < numDevices; e++) {
            SD_MAP_IMPL<ShapeDescriptor, ConstantShapeBuffer> cache;
            _cache[e] = cache;
        }
    }

    ConstantShapeHelper& ConstantShapeHelper::getInstance() {
      static ConstantShapeHelper instance;
      return instance;
    }

    ConstantShapeBuffer& ConstantShapeHelper::bufferForShapeInfo(sd::DataType dataType, char order, const std::vector<sd::LongType> &shape) {
        ShapeDescriptor descriptor(dataType, order, shape);
        return bufferForShapeInfo(descriptor);
    }

ConstantShapeBuffer& ConstantShapeHelper::bufferForShapeInfo(const sd::DataType dataType, const char order, const int rank, const sd::LongType* shape) {
        ShapeDescriptor descriptor(dataType, order, shape, rank);
        return bufferForShapeInfo(descriptor);
    }

ConstantShapeBuffer&  ConstantShapeHelper::bufferForShapeInfo(const ShapeDescriptor &descriptor) {
        int deviceId = AffinityManager::currentDeviceId();

        std::lock_guard<std::mutex> lock(_mutex);

        if (_cache[deviceId].count(descriptor) == 0) {
          auto hPtr = std::make_shared<PointerWrapper>(descriptor.toShapeInfo(), std::make_shared<PrimaryPointerDeallocator>());
          auto dPtr = std::make_shared<PointerWrapper>(ConstantHelper::getInstance().replicatePointer(hPtr->pointer(), shape::shapeInfoByteLength(hPtr->pointerAsT<sd::LongType>())), std::make_shared<CudaPointerDeallocator>());
          ConstantShapeBuffer buffer(hPtr, dPtr);
          ShapeDescriptor descriptor1(descriptor);
          _cache[deviceId][descriptor1] = buffer;
          return _cache[deviceId][descriptor1];
        } else {
          return _cache[deviceId].at(descriptor);
        }
    }

ConstantShapeBuffer&  ConstantShapeHelper::bufferForShapeInfo(const sd::LongType *shapeInfo) {
        ShapeDescriptor descriptor(shapeInfo);
        return bufferForShapeInfo(descriptor);
    }

    bool ConstantShapeHelper::checkBufferExistenceForShapeInfo(ShapeDescriptor &descriptor) {
        auto deviceId = AffinityManager::currentDeviceId();
        std::lock_guard<std::mutex> lock(_mutex);

        return _cache[deviceId].count(descriptor) != 0;
    }

    sd::LongType const* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order, const int rank, const sd::LongType* shape) {
        ShapeDescriptor descriptor(dataType, order, shape, rank);
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const sd::LongType* shapeInfo) {
        return ConstantShapeHelper::createShapeInfo(dataType, shape::order(shapeInfo), shape::rank(shapeInfo), shape::shapeOf(const_cast<sd::LongType*>(shapeInfo)));
    }

    sd::LongType const* ConstantShapeHelper::emptyShapeInfo(const sd::DataType dataType) {
        auto descriptor = ShapeDescriptor::emptyDescriptor(dataType);
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::scalarShapeInfo(const sd::DataType dataType) {
        auto descriptor = ShapeDescriptor::scalarDescriptor(dataType);
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::vectorShapeInfo(const sd::LongType length, const sd::DataType dataType) {
        auto descriptor = ShapeDescriptor::vectorDescriptor(length, dataType);
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::createShapeInfo(const sd::DataType dataType, const char order, const std::vector<sd::LongType> &shape) {
        ShapeDescriptor descriptor(dataType, order, shape);
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::createShapeInfo(const ShapeDescriptor &descriptor) {
        return bufferForShapeInfo(descriptor).primary();
    }

    sd::LongType const* ConstantShapeHelper::createFromExisting(sd::LongType *shapeInfo, bool destroyOriginal) {
        ShapeDescriptor descriptor(shapeInfo);
        auto result = createShapeInfo(descriptor);

        if (destroyOriginal)
            RELEASE(shapeInfo, nullptr);

        return result;
    }

    sd::LongType const* ConstantShapeHelper::createFromExisting(sd::LongType *shapeInfo, sd::memory::Workspace *workspace) {
        ShapeDescriptor descriptor(shapeInfo);
        auto result = createShapeInfo(descriptor);

        RELEASE(shapeInfo, workspace);

        return result;
    }

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer&  ConstantShapeHelper::createShapeInfoWithUnitiesForBroadcast(const sd::LongType* maxShapeInfo, const sd::LongType* minShapeInfo, sd::memory::Workspace* workspace, const std::vector<int>& dimensions) {

    sd::LongType* newShapeInfo = nullptr;
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(maxShapeInfo)), sd::LongType);

    newShapeInfo[0] = shape::rank(maxShapeInfo);
    newShapeInfo[2*shape::rank(maxShapeInfo)+1] = 0;
    sd::ArrayOptions::copyDataType(newShapeInfo, minShapeInfo);                     // type
    newShapeInfo[2 * newShapeInfo[0] + 2] = shape::elementWiseStride(minShapeInfo); // ews
    newShapeInfo[2 * newShapeInfo[0] + 3] = shape::order(minShapeInfo);             // order

    if(!dimensions.empty()) {

        for(sd::Unsigned k = 0, j = 0, i = 0; i < shape::rank(maxShapeInfo); ++i) {

            if(j < dimensions.size() && dimensions[j] == i) {
                shape::shapeOf(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[k];
                shape::stride(newShapeInfo)[i]  = shape::stride(minShapeInfo)[k++];
                ++j;
            }
            else{
                shape::shapeOf(newShapeInfo)[i] = 1;
                shape::stride(newShapeInfo)[i]  = 0;
                if(shape::sizeAt(minShapeInfo, k) == 1 && dimensions.size() != shape::rank(minShapeInfo))
                    ++k;
            }
        }
    }
    else{

        for(int j = shape::rank(minShapeInfo) - 1, i = shape::rank(maxShapeInfo) - 1; i >=0 ; --i) {

            if(j >= 0) {
                shape::shapeOf(newShapeInfo)[i] = shape::shapeOf(minShapeInfo)[j];
                shape::stride(newShapeInfo)[i]  = shape::shapeOf(minShapeInfo)[j] == 1 ? 0 : shape::stride(minShapeInfo)[j];
                --j;
            }
            else {
                shape::shapeOf(newShapeInfo)[i] = 1;
                shape::stride(newShapeInfo)[i]  = 0;
            }
        }
    }

    ShapeDescriptor descriptor(newShapeInfo);

    RELEASE(newShapeInfo, workspace);

    return bufferForShapeInfo(descriptor);
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer& ConstantShapeHelper::createShapeInfoWithNoUnitiesForReduce(const sd::LongType* inShapeInfo, const std::vector<int> &dimsWithUnities, sd::memory::Workspace* workspace) {

    sd::LongType* newShapeInfo = nullptr;
    ALLOCATE(newShapeInfo, workspace, shape::shapeInfoLength(shape::rank(inShapeInfo) - dimsWithUnities.size()), sd::LongType);

    int temp;
    if(dimsWithUnities.size() == 1 && shape::isCommonVector(inShapeInfo, temp) && temp == dimsWithUnities[0]) {
        auto dims = ShapeUtils::evalDimsToExclude(shape::rank(inShapeInfo), {temp});
        shape::excludeUnitiesFromShapeInfo(inShapeInfo, dims.data(), dims.size(), newShapeInfo);
    } else {
        shape::excludeUnitiesFromShapeInfo(inShapeInfo, dimsWithUnities.data(), dimsWithUnities.size(), newShapeInfo);
    }

    ShapeDescriptor descriptor(newShapeInfo);

    RELEASE(newShapeInfo, workspace);

    return bufferForShapeInfo(descriptor);
}

////////////////////////////////////////////////////////////////////////
ConstantShapeBuffer& ConstantShapeHelper::createSubArrShapeInfo(const sd::LongType* inShapeInfo, const int* dims, const int dimsSize, sd::memory::Workspace* workspace) {

    sd::LongType* newShapeInfo = ShapeBuilders::createSubArrShapeInfo(inShapeInfo, dims, dimsSize, workspace);

    ShapeDescriptor descriptor(newShapeInfo);

    RELEASE(newShapeInfo, workspace);

    return bufferForShapeInfo(descriptor);
}


}
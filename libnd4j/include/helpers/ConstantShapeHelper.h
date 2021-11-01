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

#ifndef DEV_TESTS_CONSTANTSHAPEHELPER_H
#define DEV_TESTS_CONSTANTSHAPEHELPER_H


#include <map>
#include <mutex>
#include <vector>
#include <array/ShapeDescriptor.h>
#include <array/ConstantShapeBuffer.h>
#include <memory/Workspace.h>
#include <system/op_boilerplate.h>

namespace sd {

    class SD_LIB_EXPORT ConstantShapeHelper {
    private:
        std::mutex _mutex;
        std::vector<SD_MAP_IMPL<ShapeDescriptor, ConstantShapeBuffer>> _cache;


        ConstantShapeHelper();
    public:
        ~ConstantShapeHelper() = default;

        static ConstantShapeHelper & getInstance();


        ConstantShapeBuffer& bufferForShapeInfo(sd::DataType dataType, char order, const std::vector<sd::LongType> &shape);
        ConstantShapeBuffer& bufferForShapeInfo(const ShapeDescriptor &descriptor);
        ConstantShapeBuffer& bufferForShapeInfo(const sd::LongType *shapeInfo);
        ConstantShapeBuffer& bufferForShapeInfo(sd::DataType dataType, char order, int rank, const sd::LongType* shape);
        ConstantShapeBuffer& createShapeInfoWithUnitiesForBroadcast(const sd::LongType* maxShapeInfo, const sd::LongType* minShapeInfo, sd::memory::Workspace* workspace = nullptr, const std::vector<int> &dimensions = {});
        ConstantShapeBuffer& createShapeInfoWithNoUnitiesForReduce(const sd::LongType* maxShapeInfo, const std::vector<int> &dimsWithUnities, sd::memory::Workspace* workspace = nullptr);
        ConstantShapeBuffer& createSubArrShapeInfo(const sd::LongType* inShapeInfo, const int* dims, const int dimsSize, sd::memory::Workspace* workspace = nullptr);


        const sd::LongType* emptyShapeInfo(sd::DataType dataType);
        const sd::LongType* scalarShapeInfo(sd::DataType dataType);
        const sd::LongType* vectorShapeInfo(sd::LongType length, sd::DataType dataType);
        const sd::LongType* createShapeInfo(const ShapeDescriptor &descriptor);
        const sd::LongType* createShapeInfo(sd::DataType dataType, char order, const std::vector<sd::LongType> &shape);
        const sd::LongType* createShapeInfo(sd::DataType dataType, char order, int rank, const sd::LongType* shape);
        const sd::LongType* createShapeInfo(sd::DataType dataType, const sd::LongType* shapeInfo);

        const sd::LongType* createFromExisting(sd::LongType *shapeInfo, sd::memory::Workspace *workspace);
        const sd::LongType* createFromExisting(sd::LongType *shapeInfo, bool destroyOriginal = true);

        bool checkBufferExistenceForShapeInfo(ShapeDescriptor &descriptor);


        /**
         * This method returns number of cached TAD shapes/offsets on specific device
         * @return
         */
        SD_INLINE int cachedEntriesForDevice(int deviceId) {
            if (deviceId > _cache.size())
                throw std::runtime_error("deviceId > number of actual devices");

            return _cache[deviceId].size();
        }

        /**
         * This method returns total number of cached TAD shapes/offsets on all devices
         * @return
         */
        SD_INLINE int totalCachedEntries() {
            int total = 0;

            for (int e = 0; e < _cache.size(); e++)
                total += _cache[e].size();

            return total;
        }
    };
}

#endif //DEV_TESTS_CONSTANTSHAPEHELPER_H

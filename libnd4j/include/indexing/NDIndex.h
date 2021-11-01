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

#ifndef LIBND4J_NDINDEX_H
#define LIBND4J_NDINDEX_H

#include <vector>
#include <system/common.h>

namespace sd {
    class SD_LIB_EXPORT NDIndex {
    protected:
        std::vector<sd::LongType> _indices;
        sd::LongType _stride = 1;
    public:
        NDIndex() = default;
        ~NDIndex() = default;

        bool isAll();
        bool isPoint();
        virtual bool isInterval();

        std::vector<sd::LongType>& getIndices();
        sd::LongType stride();

        static NDIndex* all();
        static NDIndex* point(sd::LongType pt);
        static NDIndex* interval(sd::LongType start, sd::LongType end, sd::LongType stride = 1);
    };

    class SD_LIB_EXPORT NDIndexAll : public NDIndex {
    public:
        NDIndexAll();
        virtual bool isInterval();
        ~NDIndexAll() = default;
    };


    class SD_LIB_EXPORT NDIndexPoint : public NDIndex {
    public:
        NDIndexPoint(sd::LongType point);
        virtual bool isInterval();
        ~NDIndexPoint() = default;
    };

    class SD_LIB_EXPORT NDIndexInterval : public NDIndex {
    public:
        NDIndexInterval(sd::LongType start, sd::LongType end, sd::LongType stride = 1);
        virtual bool isInterval();
        ~NDIndexInterval() = default;
    };
}


#endif //LIBND4J_NDINDEX_H

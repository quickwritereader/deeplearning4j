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
#include "../TadPack.h"
#include <system/Environment.h>
#include <helpers/shape.h>

namespace sd {
    TadPack::TadPack(const ConstantShapeBuffer &shapes, const ConstantOffsetsBuffer &offets, sd::LongType numTads) : _tadShape(shapes), _tadOffsets(offets) {
        _numTads = numTads;
    }

    const sd::LongType* TadPack::primaryShapeInfo() const {
        return _tadShape.primary();
    }

    const sd::LongType* TadPack::primaryOffsets() const {
        return _tadOffsets.primary();
    }

    const sd::LongType* TadPack::specialShapeInfo() const {
        return _tadShape.special();
    }

    const sd::LongType* TadPack::specialOffsets() const {
        return _tadOffsets.special();
    }

    sd::LongType TadPack::numberOfTads() const {
        return _numTads;
    }

    const sd::LongType* TadPack::platformShapeInfo() const {
        return sd::Environment::getInstance().isCPU() ? primaryShapeInfo() : specialShapeInfo();
    }

    const sd::LongType* TadPack::platformOffsets() const {
        return sd::Environment::getInstance().isCPU() ? primaryOffsets() : specialOffsets();
    }

    int TadPack::shapeInfoLength() const {
        return (int) shape::shapeInfoLength(primaryShapeInfo());
    }
}
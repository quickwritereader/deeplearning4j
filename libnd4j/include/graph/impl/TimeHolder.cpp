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
// Created by raver119 on 16/11/17.
//
#include <graph/TimeHolder.h>

namespace sd {
    namespace graph {

        void TimeHolder::setOuterTime(int nodeId, sd::LongType time) {
            _outer[nodeId] = time;
        }

        void TimeHolder::setInnerTime(int nodeId, sd::LongType time) {
            _inner[nodeId] = time;
        }

        sd::LongType TimeHolder::outerTime(int nodeId) {
            if (_outer.count(nodeId) == 0)
                return 0;

            return _outer[nodeId];
        }

        sd::LongType TimeHolder::innerTime(int nodeId) {
            if (_inner.count(nodeId) == 0)
                return 0;

            return _inner[nodeId];
        }
    }
}

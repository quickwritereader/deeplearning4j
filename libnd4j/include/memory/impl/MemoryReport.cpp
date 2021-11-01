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
// Created by raver119 on 11.10.2017.
//
#include "memory/MemoryReport.h"

bool sd::memory::MemoryReport::operator<(const sd::memory::MemoryReport &other) const {
    return this->_rss < other._rss;
}

bool sd::memory::MemoryReport::operator>(const sd::memory::MemoryReport &other) const {
    return this->_rss > other._rss;
}

bool sd::memory::MemoryReport::operator==(const sd::memory::MemoryReport &other) const {
    return this->_rss == other._rss;
}

bool sd::memory::MemoryReport::operator!=(const sd::memory::MemoryReport &other) const {
    return this->_rss != other._rss;
}

bool sd::memory::MemoryReport::operator<=(const sd::memory::MemoryReport &other) const {
    return this->_rss <= other._rss;
}

bool sd::memory::MemoryReport::operator>=(const sd::memory::MemoryReport &other) const {
    return this->_rss >= other._rss;
}

sd::LongType sd::memory::MemoryReport::getVM() const {
    return _vm;
}

void sd::memory::MemoryReport::setVM(sd::LongType _vm) {
    MemoryReport::_vm = _vm;
}

sd::LongType sd::memory::MemoryReport::getRSS() const {
    return _rss;
}

void sd::memory::MemoryReport::setRSS(sd::LongType _rss) {
    MemoryReport::_rss = _rss;
}

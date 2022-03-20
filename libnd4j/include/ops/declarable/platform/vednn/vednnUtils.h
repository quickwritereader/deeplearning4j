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

#ifndef DEV_TESTSVEDNNUTILS_H
#define DEV_TESTSVEDNNUTILS_H

#include <vednn.h>
#include <array/NDArray.h>
#include <graph/Context.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#if defined(HAVE_VEDA)
#include "veda_helper.h"
#endif
using namespace samediff;

namespace sd {
namespace ops {
namespace platforms {

/**
 * forward, backward
 */
DECLARE_PLATFORM(relu, ENGINE_CPU);
DECLARE_PLATFORM(relu_bp, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d, ENGINE_CPU);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CPU);
DECLARE_PLATFORM(conv2d, ENGINE_CPU);
DECLARE_PLATFORM(conv2d_bp, ENGINE_CPU);

//only forward
DECLARE_PLATFORM(matmul, ENGINE_CPU);
DECLARE_PLATFORM(softmax, ENGINE_CPU);
DECLARE_PLATFORM(log_softmax, ENGINE_CPU);


}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // DEV_TESTSVEDNNUTILS_H

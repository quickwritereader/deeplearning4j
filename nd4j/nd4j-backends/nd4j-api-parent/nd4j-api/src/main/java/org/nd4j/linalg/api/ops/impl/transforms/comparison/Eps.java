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

package org.nd4j.linalg.api.ops.impl.transforms.comparison;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformBoolOp;

import java.util.List;

public class Eps extends BaseTransformBoolOp {
    public Eps(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2) {
        super(sameDiff, i_v1, i_v2);
    }

    public Eps(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, boolean inPlace) {
        super(sameDiff, i_v1, i_v2, inPlace);
    }

    public Eps(SameDiff sameDiff) {
        super(sameDiff);
    }
    public Eps(INDArray indArray) {
        super(indArray);
    }


    public Eps(SameDiff sameDiff, SDVariable i_v1, SDVariable i_v2, Object[] extraArgs) {
        super(sameDiff, i_v1, i_v2, extraArgs);
    }

    public Eps() {}

    public Eps(INDArray x, INDArray y, INDArray z) {
        super(x, y, z);
    }

    @Override
    public Type getOpType() {
        return opType();
    }

    @Override
    public Type opType() {
        return Type.PAIRWISE_BOOL;
    }

    @Override
    public int opNum() {
        return 3;
    }

    @Override
    public String opName() {
        return "eps";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }
}

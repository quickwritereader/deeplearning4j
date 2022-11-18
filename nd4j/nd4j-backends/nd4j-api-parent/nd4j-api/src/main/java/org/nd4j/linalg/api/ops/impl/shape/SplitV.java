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

package org.nd4j.linalg.api.ops.impl.shape;

import lombok.val;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * SplitV op
 */
public class SplitV extends DynamicCustomOp {

    private int numSplit;
    private int splitDim;

    public SplitV() {
    }

    public SplitV(SameDiff sd, SDVariable input, SDVariable sizes, int numSplit, int splitDim) {
        super(null,sd,new SDVariable[]{input,sizes},false);
        this.splitDim = splitDim;
        this.numSplit = numSplit;
        addIArgument(splitDim,numSplit);
    }

    public SplitV(INDArray input, INDArray sizes, int numSplit, int splitDim) {
        super(null,new INDArray[]{input,sizes},null);
        this.numSplit = numSplit;
        this.splitDim = splitDim;
        addIArgument(splitDim,numSplit);
    }

    @Override
    public String opName() {
        return "split_v";
    }

    @Override
    public String tensorflowName() {
        return "SplitV";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val splitDim = TFGraphMapper.getArrayFrom(TFGraphMapper.getNodeWithNameFromGraph(graph,nodeDef.getInput(0)),graph);
        if(splitDim != null) {
            this.splitDim = splitDim.getInt(0);
            addIArgument(splitDim.getInt(0));
        }

        val numSplits = (int) attributesForNode.get("num_split").getI();
        this.numSplit = numSplits;
        //addIArgument(numSplits);  //libnd4j op doesn't used/need it for execution
    }

    @Override
    public Map<String, Object> propertiesForFunction() {
        Map<String,Object> ret = new HashMap<>();
        ret.put("numSplit",numSplit);
        ret.put("splitDim",splitDim);
        return ret;
    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String,Map<String,PropertyMapping>> ret = new HashMap<>();
        Map<String,PropertyMapping> map = new HashMap<>();

        val splitDim = PropertyMapping.builder()
                .tfInputPosition(-1)
                .propertyNames(new String[]{"splitDim"})
                .build();

        val numSplit = PropertyMapping.builder()
                .tfAttrName("num_split")
                .propertyNames(new String[]{"numSplit"})
                .build();

        map.put("numSplit",numSplit);
        map.put("splitDim",splitDim);

        ret.put(tensorflowName(),map);
        //ret.put(onnxName(),map);

        return ret;
    }

    @Override
    public void configureWithSameDiff(SameDiff sameDiff) {
        super.configureWithSameDiff(sameDiff);
    }

    @Override
    public void configureFromArguments() {
        super.configureFromArguments();
    }

    @Override
    public void setPropertiesForFunction(Map<String, Object> properties) {
        if(properties.containsKey("splitDim")) {
            Integer splitDim = getIntValueFromProperty("splitDim",properties);
            this.splitDim = splitDim;
        }

        if(properties.containsKey("numSplit")) {
            Integer numSplit = getIntValueFromProperty("numSplit",properties);
            this.numSplit = numSplit;
        }
    }

    @Override
    public int getNumOutputs() {
        return numSplit;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(new Concat(sameDiff,splitDim,f1.toArray(new SDVariable[f1.size()])).outputVariables());
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> dataTypes) {
        //Output types are same as first input type - just numSplits of them...
        List<DataType> out = new ArrayList<>(numSplit);
        for( int i = 0; i < numSplit; i++) {
            out.add(dataTypes.get(0));
        }
        return out;
    }

}

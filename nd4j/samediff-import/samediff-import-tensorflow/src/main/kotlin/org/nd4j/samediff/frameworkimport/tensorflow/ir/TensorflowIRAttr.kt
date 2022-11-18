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
package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.nd4j.ir.TensorNamespace
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.tensorflow.framework.AttrValue
import org.tensorflow.framework.DataType
import org.tensorflow.framework.OpDef
import org.tensorflow.framework.TensorProto

class TensorflowIRAttr(inputAttributeDef: OpDef.AttrDef, inputAttributeValue: AttrValue):
    IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.list.fList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        if(attributeDef.type == "shape") {
            return attributeValue.shape.dimList.toList().map { input -> input.size }
        }
        else
            return attributeValue.list.iList
    }

    override fun boolValue(): Boolean {
        return attributeValue.b
    }

    override fun listBoolValue(): List<Boolean> {
        return attributeValue.list.bList
    }

    override fun shapeValue(): List<Long> {
        return attributeValue.shape.dimList.map { input -> input.size }
    }

    override fun listDataTypes(): List<TensorNamespace.DataType> {
        throw UnsupportedOperationException("Unable to map list of data types")
    }

    override fun attributeValueType(): AttributeValueType {
        when(attributeDef.type) {
            "shape" -> return AttributeValueType.SHAPE
            "list(bool)" -> return AttributeValueType.LIST_BOOL
            "bool" -> return AttributeValueType.BOOL
            "string" -> return AttributeValueType.STRING
            "list(string)" -> return AttributeValueType.LIST_STRING
            "int" -> return AttributeValueType.INT
            "list(int)" -> return AttributeValueType.LIST_INT
            "float" -> return AttributeValueType.FLOAT
            "list(float)" -> return AttributeValueType.LIST_FLOAT
            "tensor" -> return AttributeValueType.TENSOR
            "list(tensor)" -> return AttributeValueType.LIST_TENSOR
            "type" -> return AttributeValueType.DATA_TYPE
            "list(type)" -> return AttributeValueType.LIST_DATA_TYPE
        }

        return AttributeValueType.INVALID
    }



    override fun internalAttributeDef(): OpDef.AttrDef {
        return attributeDef
    }

    override fun internalAttributeValue(): AttrValue {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<TensorProto, DataType>> {
        return attributeValue.list.tensorList.map { input ->
            TensorflowIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<TensorProto, DataType> {
        return TensorflowIRTensor(attributeValue.tensor)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.list.sList.map { it.toStringUtf8() }
    }

    override fun dataTataTypeValue(): IRDataType<DataType> {
        return TensorflowIRDataType(attributeValue.type)
    }

    override fun graphValue(registry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>): IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum> {
        throw UnsupportedOperationException("Unsupported for Tensorflow. Graphs do not exist on attributes.")
    }

    override fun listGraphValue(registry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>): List<IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>> {
        throw UnsupportedOperationException("Unsupported for Tensorflow. Graphs do not exist on attributes.")
    }

}
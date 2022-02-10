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

package org.nd4j.graph;

import java.nio.*;
import java.lang.*;
import java.util.*;
import com.google.flatbuffers.*;

@SuppressWarnings("unused")
public final class IntTriple extends Table {
  public static void ValidateVersion() { Constants.FLATBUFFERS_1_12_0(); }
  public static IntTriple getRootAsIntTriple(ByteBuffer _bb) { return getRootAsIntTriple(_bb, new IntTriple()); }
  public static IntTriple getRootAsIntTriple(ByteBuffer _bb, IntTriple obj) { _bb.order(java.nio.ByteOrder.LITTLE_ENDIAN); return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __reset(_i, _bb); }
  public IntTriple __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int first() { int o = __offset(4); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public int second() { int o = __offset(6); return o != 0 ? bb.getInt(o + bb_pos) : 0; }
  public int third() { int o = __offset(8); return o != 0 ? bb.getInt(o + bb_pos) : 0; }

  public static int createIntTriple(FlatBufferBuilder builder,
      int first,
      int second,
      int third) {
    builder.startTable(3);
    IntTriple.addThird(builder, third);
    IntTriple.addSecond(builder, second);
    IntTriple.addFirst(builder, first);
    return IntTriple.endIntTriple(builder);
  }

  public static void startIntTriple(FlatBufferBuilder builder) { builder.startTable(3); }
  public static void addFirst(FlatBufferBuilder builder, int first) { builder.addInt(0, first, 0); }
  public static void addSecond(FlatBufferBuilder builder, int second) { builder.addInt(1, second, 0); }
  public static void addThird(FlatBufferBuilder builder, int third) { builder.addInt(2, third, 0); }
  public static int endIntTriple(FlatBufferBuilder builder) {
    int o = builder.endTable();
    return o;
  }

  public static final class Vector extends BaseVector {
    public Vector __assign(int _vector, int _element_size, ByteBuffer _bb) { __reset(_vector, _element_size, _bb); return this; }

    public IntTriple get(int j) { return get(new IntTriple(), j); }
    public IntTriple get(IntTriple obj, int j) {  return obj.__assign(__indirect(__element(j), bb), bb); }
  }
}


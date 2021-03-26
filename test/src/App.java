import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.custom.Polygamma;
import org.nd4j.linalg.api.ops.impl.shape.StridedSlice;
import org.nd4j.linalg.api.ops.impl.transforms.custom.BatchToSpaceND;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.memory.abstracts.Nd4jWorkspace; 
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend; 
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream; 
import java.util.concurrent.CopyOnWriteArrayList;
/**
 * Hello world!
 *
 */
public class App 
{

        private static final WorkspaceConfiguration basicConfiguration = WorkspaceConfiguration.builder().initialSize(81920)
        .overallocationLimit(0.1).policySpill(SpillPolicy.EXTERNAL).policyLearning(LearningPolicy.NONE)
        .policyMirroring(MirroringPolicy.FULL).policyAllocation(AllocationPolicy.OVERALLOCATE).build();


public static void assertEquals(Object x, Object y){
        if(x.equals(y)){
                System.out.println("ok");
        }else{
                System.out.println(" differen "+x.toString() + " " + y.toString());
        }
}

public static void assertNull(Object y){
        if(y == null){
                System.out.println("ok");
        }else{
                System.out.println(" not null");
        }    
}

public static void assertTrue(boolean y){
        if(y == true){
                System.out.println("ok");
        }else{
                System.out.println(" not true");
        }    
}

 public static void testX(){
        Nd4j.getWorkspaceManager().setDefaultWorkspaceConfiguration(basicConfiguration);
        try (Nd4jWorkspace ws1 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1").notifyScopeEntered()) {
            INDArray array = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
            long reqMem = 5 * array.dataType().width();
            long add = ((Nd4jWorkspace.alignmentBase) - reqMem % (Nd4jWorkspace.alignmentBase));
            assertEquals(reqMem + add, ws1.getPrimaryOffset());
            try (Nd4jWorkspace ws2 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS2").notifyScopeEntered()) {
                INDArray array2 = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
                reqMem = 5 * array2.dataType().width();
                assertEquals(reqMem + ((Nd4jWorkspace.alignmentBase) - reqMem % (Nd4jWorkspace.alignmentBase)), ws1.getPrimaryOffset());
                assertEquals(reqMem + ((Nd4jWorkspace.alignmentBase) - reqMem % (Nd4jWorkspace.alignmentBase)), ws2.getPrimaryOffset());
                try (Nd4jWorkspace ws3 = (Nd4jWorkspace) Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("WS1")
                        .notifyScopeBorrowed()) {
                    assertTrue(ws1 == ws3);
                    INDArray array3 = Nd4j.create(new float[] {1f, 2f, 3f, 4f, 5f});
                    assertEquals(reqMem + ((Nd4jWorkspace.alignmentBase) - reqMem % (Nd4jWorkspace.alignmentBase)), ws2.getPrimaryOffset());
                    assertEquals((reqMem + ((Nd4jWorkspace.alignmentBase) - reqMem % (Nd4jWorkspace.alignmentBase))) * 2, ws1.getPrimaryOffset());
                }
            }
        }
        assertNull(Nd4j.getMemoryManager().getCurrentWorkspace());
 }

 
    public static void main(String[] args)
    { 
            testX();
//         INDArray n = Nd4j.linspace(DataType.DOUBLE, 1.0, 1.0, 9).reshape(3,3);
//         INDArray x = Nd4j.create(DataType.DOUBLE, 3,3);
//         x.assign(0.5);  
//         INDArray expected = Nd4j.createFromArray(new double[]{4.934802, -16.828796, 97.409088, -771.474243,
//                 7691.113770f, -92203.460938f, 1290440.250000, -20644900.000000, 3.71595e+08}).reshape(3,3);
//         INDArray output = Nd4j.create(DataType.DOUBLE, expected.shape());
//         var op = new Polygamma(x,n,output);
//         Nd4j.exec(op);
//         System.out.println(output);

//          var opx = DynamicCustomOp.builder("polygamma").addInputs(x,n).build();

// Nd4j.getExecutioner().execAndReturn(opx);

// System.out.println(output);

// var ogp = new Polygamma(n,x);
// var ret= Nd4j.exec(ogp);
// System.out.println(ret[0]);


// INDArray out = Nd4j.create(DataType.FLOAT, 2, 4, 5);
// var t=Nd4j.createFromArray(1, 2);
// System.out.println(t);
// System.out.println(t.shapeInfoToString());
// var tt=Nd4j.createFromArray(new int[][]{ new int[]{0, 0}, new int[]{0, 1} });
// System.out.println(tt);
// System.out.println(tt.shapeInfoToString());
// var xx =Nd4j.rand(DataType.FLOAT, new int[]{4, 4, 3});
// System.out.println(xx);
// DynamicCustomOp c = new BatchToSpaceND();

// c.addInputArgument(
//         xx,
//         Nd4j.createFromArray(1, 2),
//         Nd4j.createFromArray(new int[][]{ new int[]{0, 0}, new int[]{0, 1} })
// );
// c.addOutputArgument(out);
// Nd4j.getExecutioner().exec(c);
// List<LongShapeDescriptor> l = c.calculateOutputShape();
//         System.out.println(Arrays.toString(l.get(0).getShape()));
//          System.out.println(out);
   }
}

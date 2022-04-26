package com.javatensormath.core;

import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Sub;
import org.tensorflow.types.TFloat32;

import java.util.HashMap;
import java.util.Map;

public class App {

    public static void main(String[] args){

        FloatDataBuffer flBuffer1 = DataBuffers.of(1f, 2f, 3f, 4f);
        FloatDataBuffer flBuffer2 = DataBuffers.of(5f, 6f, 7f, 8f);

        FloatNdArray data1 = NdArrays.wrap(Shape.of(2, 2), flBuffer1);
        FloatNdArray data2 = NdArrays.wrap(Shape.of(2, 2), flBuffer2);
        Shape shape = Shape.of(2, 2);

        try(Tensor tensor = Tensor.of(TFloat32.class, shape, data1::copyTo);
            Tensor tensor1 = Tensor.of(TFloat32.class, shape, data2::copyTo);
            ConcreteFunction multiplyTensor = ConcreteFunction.create(App::multiply);
            ConcreteFunction subtractTensor = ConcreteFunction.create(App::subtract);
            ConcreteFunction divideTensor = ConcreteFunction.create(App::divide);
            ConcreteFunction addTensor = ConcreteFunction.create(App::sum)){

            Map<String, Tensor> inputs = new HashMap<>();

            //It is important to note that the keys must have the same name as the name on the
            //Signature input on the function that will perform the operation
            inputs.put("X", tensor);
            inputs.put("Y", tensor1);

            Map<String, Tensor> resultMult = multiplyTensor.call(inputs);
            Map<String, Tensor> resultSubt = subtractTensor.call(inputs);
            Map<String, Tensor> resultDivi = divideTensor.call(inputs);
            Map<String, Tensor> resultSum = addTensor.call(inputs);

            ((TFloat32) resultMult.get("Result")).elements(1).forEach(x -> System.out.print(x.getFloat() + "\t"));
            System.out.println();
            ((TFloat32) resultSubt.get("Result")).elements(1).forEach(x -> System.out.print(x.getFloat() + "\t"));
            System.out.println();
            ((TFloat32) resultDivi.get("Result1")).elements(1).forEach(x -> System.out.print(x.getFloat() + "\t"));
            System.out.println();
            ((TFloat32) resultDivi.get("Result2")).elements(1).forEach(x -> System.out.print(x.getFloat() + "\t"));
            System.out.println();
            ((TFloat32) resultSum.get("Result")).elements(1).forEach(x -> System.out.print(x.getFloat() + "\t"));

            resultMult.get("Result").close();
            resultSubt.get("Result").close();
            resultDivi.get("Result1").close();
            resultDivi.get("Result2").close();

        }

    }

    private static Signature multiply(Ops mul){
        Placeholder<TFloat32> x = mul.placeholder(TFloat32.class);
        Placeholder<TFloat32> y = mul.placeholder(TFloat32.class);
        Mul<TFloat32> mult = mul.math.mul(x, y);

        return Signature.builder().input("X", x)
                .input("Y", y)
                .output("Result", mult)
                .build();
    }

    private static Signature subtract(Ops sub){
        Placeholder<TFloat32> x = sub.placeholder(TFloat32.class);
        Placeholder<TFloat32> y = sub.placeholder(TFloat32.class);

        Sub<TFloat32> subtract = sub.math.sub(x, y);

        return Signature.builder()
                .input("X", x)
                .input("Y", y)
                .output("Result", subtract)
                .build();
    }

    private static Signature divide(Ops div){
        Placeholder<TFloat32> x = div.placeholder(TFloat32.class);
        Placeholder<TFloat32> y = div.placeholder(TFloat32.class);

        Div<TFloat32> divide1 = div.math.div(x, y);
        Div<TFloat32> divide2 = div.math.div(y, x);

        return Signature.builder()
                .input("X", x)
                .input("Y", y)
                .output("Result1", divide1)
                .output("Result2", divide2)
                .build();
    }

    private static Signature sum(Ops sum){
        Placeholder<TFloat32> x = sum.placeholder(TFloat32.class);
        Placeholder<TFloat32> y = sum.placeholder(TFloat32.class);

        Add<TFloat32> add = sum.math.add(x, y);

        return Signature.builder()
                .input("X", x)
                .input("Y", y)
                .output("Result", add)
                .build();
    }
}

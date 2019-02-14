package com.github.packtpub.cntkquickstart;

import com.microsoft.CNTK.*;

public class FlowerClassifier {
    private final DeviceDescriptor deviceDescriptor;
    private final Function modelFunction;

    private FlowerClassifier(Function modelFunction, DeviceDescriptor deviceDescriptor) {
        this.deviceDescriptor = deviceDescriptor;
        this.modelFunction = modelFunction;
    }

    public float[] predict(float petalWidth, float petalLength, float sepalWidth, float sepalLength) {
        Variable features = modelFunction.getInputs().get(0);
        Variable output = modelFunction.getOutputs().get(0);

        FloatVectorVector batch = new FloatVectorVector();

        batch.add(createFloatVector(sepalLength, sepalWidth, petalLength, petalWidth));

        UnorderedMapVariableValuePtr inputMapping = new UnorderedMapVariableValuePtr();
        UnorderedMapVariableValuePtr outputMapping = new UnorderedMapVariableValuePtr();

        inputMapping.add(features, Value.createDenseFloat(features.getShape(), batch, deviceDescriptor));
        outputMapping.add(output, null);

        modelFunction.evaluate(inputMapping, outputMapping, deviceDescriptor);

        FloatVectorVector outputBuffer = new FloatVectorVector();
        outputMapping.getitem(output).copyVariableValueToFloat(output, outputBuffer);

        FloatVector predictedOutput = outputBuffer.get(0);

        return getVectorValues(predictedOutput);
    }

    private static float[] getVectorValues(FloatVector predictedOutput) {
        float[] outputVector = new float[(int) predictedOutput.size()];

        for (int i = 0; i < predictedOutput.size(); i++) {
            outputVector[i] = predictedOutput.get(i);
        }

        return outputVector;
    }

    private static FloatVector createFloatVector(float... inputValues) {
        FloatVector vector = new FloatVector();

        for (float inputValue : inputValues) {
            vector.add(inputValue);
        }

        return vector;
    }

    public static FlowerClassifier create() {
        DeviceDescriptor cpuDevice = DeviceDescriptor.getCPUDevice();
        Function modelFunction = Function.load("model.onnx", cpuDevice, ModelFormat.ONNX);

        return new FlowerClassifier(modelFunction, cpuDevice);
    }
}

using System.Collections.Generic;
using CNTK;

namespace PredictSpecies
{
    public class FlowerClassifier
    {
        private readonly DeviceDescriptor _deviceDescriptor;
        private readonly Function _modelFunction;

        private FlowerClassifier(Function modelFunction, DeviceDescriptor deviceDescriptor)
        {
            _deviceDescriptor = deviceDescriptor;
            _modelFunction = modelFunction;
        }

        public IList<float> Predict(float petalWidth, float petalLength, float sepalWidth, float sepalLength)
        {
            var features = _modelFunction.Inputs[0];
            var output = _modelFunction.Outputs[0];

            var inputMapping = new Dictionary<Variable, Value>();
            var outputMapping = new Dictionary<Variable, Value>();

            var batch = Value.CreateBatch(
                features.Shape,
                new float[] { sepalLength, sepalWidth, petalLength, petalWidth },
                _deviceDescriptor);

            inputMapping.Add(features, batch);
            outputMapping.Add(output, null);

            _modelFunction.Evaluate(inputMapping, outputMapping, _deviceDescriptor);

            var outputValues = outputMapping[output].GetDenseData<float>(output);

            return outputValues[0];
        }

        public static FlowerClassifier Create()
        {
            var deviceDescriptor = DeviceDescriptor.CPUDevice;
            var function = Function.Load("model.onnx", deviceDescriptor);

            return new FlowerClassifier(function, deviceDescriptor);
        }

        private static FloatVector CreateFloatVector(params float[] inputValues)
        {
            FloatVector vector = new FloatVector();

            foreach (var inputValue in inputValues)
            {
                vector.Add(inputValue);
            }

            return vector;
        }
    }
}
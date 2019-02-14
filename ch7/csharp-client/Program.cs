using System;
using CNTK;

namespace PredictSpecies
{
    class Program
    {
        static void Main(string[] args)
        {
            var classifier = FlowerClassifier.Create();
            var classNames = new string[] {
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-virginica"
            };

            var probabilities = classifier.Predict(2.0f, 4.3f, 0.1f, 1.0f);

            for (int i = 0; i < classNames.Length; i++)
            {
                Console.WriteLine($"{classNames[i]}: {probabilities[i]}");
            }
        }
    }
}

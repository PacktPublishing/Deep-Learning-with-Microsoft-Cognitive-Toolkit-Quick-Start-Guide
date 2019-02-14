package com.github.packtpub.cntkquickstart;

public class Program {
    public static void main(String... args) {
        FlowerClassifier classifier = FlowerClassifier.create();

        String[] classNames = new String[] {
                "Iris-setosa",
                "Iris-versicolor",
                "Iris-virginica"
        };

        float[] probabilities = classifier.predict(
                2.0f, 4.3f, 0.1f, 1.0f);

        for(int i = 0; i < classNames.length; i++) {
            System.out.println(String.format("%s: %f", classNames[i], probabilities[i]));
        }
    }
}

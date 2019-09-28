import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;
import java.io.IOException;

public class ML_Logistic_Regression {
    public static void main(String[] args) {
        String training_dataSet_Filename = "src/diabete_training.arff";
        String testing_dataSet_Filename = "src/diabete_testing.arff";
        String predict_dataSet_Filename = "src/diabete_predict.arff";

        process(training_dataSet_Filename, testing_dataSet_Filename, predict_dataSet_Filename,8);

        training_dataSet_Filename = "src/weather_training.arff";
        testing_dataSet_Filename = "src/weather_testing.arff";
        predict_dataSet_Filename = "src/weather_predict.arff";
        process(training_dataSet_Filename, testing_dataSet_Filename, predict_dataSet_Filename,4);

    }

    public static void process (String trainingFile, String testingFile, String predictFile, int classIndex) {
        Instances trainingDataSet = getDataSet(trainingFile, classIndex);
        Instances testingDataSet = getDataSet(testingFile, classIndex);

        Classifier classifier = new Logistic();
        try {
            classifier.buildClassifier(trainingDataSet);
            Evaluation eval = new Evaluation(trainingDataSet);
            eval.evaluateModel(classifier, testingDataSet);
            System.out.println("Logistics Regression Evaluation with Dataset");
            System.out.println(eval.toSummaryString());
            System.out.println("Logistics Regression Equation");
            System.out.println(classifier);
            System.out.println("Prediction");

            Instances predictDataSet = getDataSet(predictFile, classIndex);
            for (int i = 0; i < predictDataSet.numInstances(); i++) {

                if (classIndex == 8) {
                    System.out.println("Number of Pragnancy is " + predictDataSet.instance(i).value(0));
                    System.out.println("Number of Blood pressure value is " + predictDataSet.instance(i).value(2));
                    double value = classifier.classifyInstance(predictDataSet.instance(i));
                    System.out.println(value == 0 ? "You are fine." : "You will die in soon.");
                } else {
                    double outlook = predictDataSet.instance(i).value(0);
                    String outlookString = "rainy";
                    if (outlook == 0) {
                        outlookString = "sunny";
                    } else if (outlook == 1) {
                        outlookString = "overcast";
                    }

                    System.out.println("Outlook is " + outlookString);

                    double temp = predictDataSet.instance(i).value(1);
                    String tempString = "Cool";
                    if (temp == 0) {
                        tempString = "Hot";
                    } else if (temp == 1) {
                        tempString = "Mild";
                    }

                    System.out.println("Temperature is " + tempString);

                    double value = classifier.classifyInstance(predictDataSet.instance(i));
                    System.out.println(value == 0 ? "Play" : "Not yet");
                }

                System.out.println();
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Instances getDataSet(String fileName, int classIndex) {
        ArffLoader loader = new ArffLoader();
        try {
            loader.setFile(new File(fileName));
            Instances dataSet = loader.getDataSet();
            dataSet.setClassIndex(classIndex);
            return dataSet;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}

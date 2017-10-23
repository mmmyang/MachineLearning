package NB;

import java.io.File;

public class NBTextClassifier {
    public static void main(String[] args) throws MyException {
        System.out.println("parsing arguments...");
        // parse arguments
        if (args.length != 2) {
            throw new MyException("Wrong Input! Please enter the arguments as: <data-file-location> <to-filter-stopwords>");
        }        
        String dataPath = args[0];  // the path containing the test and train data
        boolean toFilterStopwords = args[1].equalsIgnoreCase("yes");
        
        String spamTrainPath = dataPath + "/train/spam";
        String hamTrainPath = dataPath + "/train/ham";
        String spamTestPath = dataPath + "/test/spam";
        String hamTestPath = dataPath + "/test/ham";
        String stopWordsPath = dataPath + "/stopwords.txt";
        
        NaiveBayesMultinomial nb = new NaiveBayesMultinomial();
        
        // NB training
        System.out.println("Naive Bayes Training...");
        nb.NBTraining(spamTrainPath, hamTrainPath, stopWordsPath, toFilterStopwords);
        
        // NB testing
        System.out.println("Naive Bayes Testing...");
        // test spam files
        System.out.println("Testing on spam files...");
        File spamTestFile = new File(spamTestPath);
        int spamTestFileTotalNum = spamTestFile.listFiles().length;
        double spamTestCorrectNum = 0.0;
        for (File file : spamTestFile.listFiles()) {
            if (nb.NBClassification(file) == 1) {
                spamTestCorrectNum ++;
            }
        }
        //System.out.println(spamTestFileTotalNum);
        double spamTestAccuracy = 100.0 * spamTestCorrectNum / spamTestFileTotalNum;
        System.out.println("Accuracy (%) of spam file testing is: " + spamTestAccuracy);
        
        // test ham files
        System.out.println("Testing on ham files...");
        File hamTestFile = new File(hamTestPath);
        int hamTestFileTotalNum = hamTestFile.listFiles().length;
        double hamTestCorrectNum = 0.0;
        for (File file : hamTestFile.listFiles()) {
            if (nb.NBClassification(file) == 0) {
                hamTestCorrectNum ++;
            }
        }
        double hamTestAccuracy = 100.0 * hamTestCorrectNum / hamTestFileTotalNum;
        System.out.println("Accuracy (%) of ham testing is: " + hamTestAccuracy);
        
        // result
        double testAccuracy = 100.0 * (spamTestCorrectNum + hamTestCorrectNum ) / (spamTestFileTotalNum + hamTestFileTotalNum);
        System.out.println();
        System.out.println("Accuracy (%) of all the testing is: " + testAccuracy);
    }
}

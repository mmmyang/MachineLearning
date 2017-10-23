package LR;

import java.io.File;

public class LRTextClassifier {
    
    public static void main(String[] args) throws MyException {
        System.out.println("parsing arguments...");
        // parse arguments
        if (args.length != 5) {
            throw new MyException("Wrong Input! Please enter the arguments as: <data-file-location> <to-filter-stopwords> <eta> <lambda> <iteration-num>");
        }
        String dataPath = args[0];  // the path containing the test and train data
        boolean toFilterStopwords = args[1].equalsIgnoreCase("yes");
        
        String spamTrainPath = dataPath + "/train/spam";
        String hamTrainPath = dataPath + "/train/ham";
        String spamTestPath = dataPath + "/test/spam";
        String hamTestPath = dataPath + "/test/ham";
        String stopWordsPath = dataPath + "/stopwords.txt";
        
        double eta = Double.parseDouble(args[2]); // learning rate
        double lambda = Double.parseDouble(args[3]);    // regularization coefficient
        int iterationNum = Integer.parseInt(args[4]);   // iteration number
        
        LogisticRegressionMCAP lr = new LogisticRegressionMCAP(eta, lambda, iterationNum);
        
        // LR training
        System.out.println("Logistic Regression Training...");
        lr.LRTraining(spamTrainPath, hamTrainPath, stopWordsPath, toFilterStopwords);
        
        // LR testing
        System.out.println("Logistic Regression Testing...");
        
        // test spam files
        System.out.println("Testing on spam files...");
        File spamTestFile = new File(spamTestPath);
        int spamTestFileTotalNum = spamTestFile.listFiles().length;
        double spamTestCorrectNum = 0.0;
        for (File file : spamTestFile.listFiles()) {
            if (lr.LRClassification(file) == 1) {
                spamTestCorrectNum ++;
            }
        }
        
        double spamTestAccuracy = 100.0 * spamTestCorrectNum / spamTestFileTotalNum;
        System.out.println("Accuracy (%) of spam file testing is: " + spamTestAccuracy);
        
        // test ham files
        System.out.println("Testing on ham files...");
        File hamTestFile = new File(hamTestPath);
        int hamTestFileTotalNum = hamTestFile.listFiles().length;
        double hamTestCorrectNum = 0.0;
        for (File file : hamTestFile.listFiles()) {
            if (lr.LRClassification(file) == 0) {
                hamTestCorrectNum ++;
            }
        }
        double hamTestAccuracy = 100.0 * hamTestCorrectNum / hamTestFileTotalNum;
        System.out.println("Accuracy (%) of ham testing is: " + hamTestAccuracy);
        
        
        //System.out.println(spamTestCorrectNum);
        //System.out.println(spamTestFileTotalNum);
        //System.out.println(hamTestCorrectNum);
        //System.out.println(hamTestFileTotalNum);
        
        // result
        double testAccuracy = 100.0 * (spamTestCorrectNum + hamTestCorrectNum ) / (spamTestFileTotalNum + hamTestFileTotalNum);
        System.out.println();
        System.out.println("Accuracy (%) of all the testing is: " + testAccuracy);
       
    }
    
}

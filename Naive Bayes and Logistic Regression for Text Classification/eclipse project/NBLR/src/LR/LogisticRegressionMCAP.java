package LR;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Set;

public class LogisticRegressionMCAP {
    private LinkedHashMap<String, LinkedHashMap<String, Integer>> wordFreqOfEachFileSpam = new LinkedHashMap<>(200);
    private LinkedHashMap<String, LinkedHashMap<String, Integer>> wordFreqOfEachFileHam = new LinkedHashMap<>(200);
    
    private Set<String> filesSpamHam = new HashSet<>();
    private Set<String> filesSpam = new HashSet<>();
    private Set<String> filesHam = new HashSet<>();
    
    private Set<String> vocabulary = new HashSet<>();
    private Set<String> stopwords = new HashSet<>();
    
    private LinkedHashMap<String, Double> weightsOfEachWord = new LinkedHashMap<>();
    private LinkedHashMap<String, Double> weightsOfEachWordTemp = new LinkedHashMap<>();
    
    private double eta = 0.0;   // learning rate
    private double lambda = 0.0;    // regularization coefficient
    private int iterationNum = 0;   // iteration number
    
    private double w0 = 0.01;
    
    /**
     * constructor
     * @param alpha
     * @param lambda
     * @param iterationNum
     */
    public LogisticRegressionMCAP(double eta, double lambda, int iterationNum) {
        this.eta = eta;
        this.lambda = lambda;
        this.iterationNum = iterationNum;
    }
    
    /**
     * count and store word frequency of words in each file, and build vocabulary at the same time
     * @param dataPath
     * @param dataClass is "spam" if the data is spam training data, and is "ham" if the data is ham training data
     */
    public void countWordFreqOfEachFile(String dataPath, String dataClass) {
        File dataFile = new File(dataPath);
        
        LinkedHashMap<String, LinkedHashMap<String, Integer>> wordFreqOfEachFile = null;
        if (dataClass.equalsIgnoreCase("spam")) {
            wordFreqOfEachFile = wordFreqOfEachFileSpam;
        } else {
            wordFreqOfEachFile = wordFreqOfEachFileHam;
        }
        
        for(File file: dataFile.listFiles()) {
            LinkedHashMap<String, Integer> wordFreqOfCurrFile = new LinkedHashMap<>();
            Scanner sc = null;
            try {
                sc = new Scanner(file);
                //sc.useDelimiter("[^\\w]");  // scan only words
                sc.useDelimiter("[^a-zA-Z]+");
                String nextWord;
                while (sc.hasNext()) {
                    nextWord = sc.next();
                    if (wordFreqOfCurrFile.containsKey(nextWord)) {
                        wordFreqOfCurrFile.put(nextWord, wordFreqOfCurrFile.get(nextWord) + 1);
                    } else {
                        wordFreqOfCurrFile.put(nextWord, 1);
                    }
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } finally {
                if (sc != null) {
                    sc.close();
                }
            }
            wordFreqOfEachFile.put(file.getName(), wordFreqOfCurrFile);
            vocabulary.addAll(wordFreqOfCurrFile.keySet()); // build vocabulary
        }
    }
    
    public void filterStopwords(String stopWordsPath) {
        // scan stopwords to a set
        Scanner sc = null;
        try {
            sc = new Scanner(new File(stopWordsPath));
            //sc.useDelimiter("[^\\w]");  // scan only words
            sc.useDelimiter("[^a-zA-Z]+");
            String nextWord;
            while (sc.hasNext()) {
                nextWord = sc.next();  
                stopwords.add(nextWord);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (sc != null) {
                sc.close();
            }
        }
        
        // filter stopwords
        for (String stopword : stopwords) {
            vocabulary.remove(stopword);
        }  
    }
    
    private double getprobOfSpamGivenVariables(String fileName, String fileClass) {
        double weightVarSum = w0;
        if (fileClass.equalsIgnoreCase("spam")) {
            for(Entry<String, Integer> wordFreqPair : wordFreqOfEachFileSpam.get(fileName).entrySet()){
                if (weightsOfEachWord.containsKey(wordFreqPair.getKey())) {
                    weightVarSum += weightsOfEachWord.get( wordFreqPair.getKey() ) * wordFreqPair.getValue();
                }  
            }
        } else {
            for(Entry<String, Integer> wordFreqPair : wordFreqOfEachFileHam.get(fileName).entrySet()){
                if (weightsOfEachWord.containsKey(wordFreqPair.getKey())) {
                    weightVarSum += weightsOfEachWord.get( wordFreqPair.getKey() ) * wordFreqPair.getValue();
                }
            }
        }
        double result;
        if (weightVarSum > 100) {
            result = 1.0;
        } else if (weightVarSum < -100) {
            result = 0.0;
        } else {
            double temp = Math.exp(weightVarSum);
            result = temp / (1.0 + temp);
        }
        return result;
    }
    
    public void calWeights() {
        // initialize weights
        double initialWeight = 0.0;
        for (String currWord : vocabulary) {
            initialWeight = Math.random() * 2 - 1;  // initial weight between -1 and 1
            weightsOfEachWord.put(currWord, initialWeight);
        }
        
        // update weights using gradient ascent
        for (int i = 1; i <= iterationNum; i++) {   // stop in fixed iteration instead of waiting convergence
            //System.out.println("iteration round " + i);
            for (String currWord : vocabulary) {    // calculate new weight for each word
                double deltaWeightError = 0.0;
                
                // deltaWeightError come from each spam file
                for (Entry<String, LinkedHashMap<String, Integer>> entry : wordFreqOfEachFileSpam.entrySet()) {
                    int currWordNumInCurrFile = 0;
                    if (entry.getValue().containsKey(currWord)) {
                        currWordNumInCurrFile = entry.getValue().get(currWord);
                    }
                    double probOfSpamGivenVariables = getprobOfSpamGivenVariables(entry.getKey(), "spam"); //!!!
                    deltaWeightError += currWordNumInCurrFile * ( 1 - probOfSpamGivenVariables );
                }
                // deltaWeightError come from each ham file
                for (Entry<String, LinkedHashMap<String, Integer>> entry : wordFreqOfEachFileHam.entrySet()) {
                    int currWordNumInCurrFile = 0;
                    if (entry.getValue().containsKey(currWord)) {
                        currWordNumInCurrFile = entry.getValue().get(currWord);
                    }
                    double probOfSpamGivenVariables = getprobOfSpamGivenVariables(entry.getKey(), "ham"); //!!!
                    deltaWeightError += currWordNumInCurrFile * ( 0 - probOfSpamGivenVariables );
                }
                double newWeightOfCurrWord = weightsOfEachWord.get(currWord) + eta * deltaWeightError - eta * lambda * weightsOfEachWord.get(currWord);
                weightsOfEachWordTemp.put(currWord, newWeightOfCurrWord);
            }
            
            // update weight for each word
            weightsOfEachWord.putAll(weightsOfEachWordTemp);
        }
        
    }
    
    public void LRTraining(String spamTrainPath, String hamTrainPath, String stopWordsPath, boolean toFilterStopwords) {
        countWordFreqOfEachFile(spamTrainPath, "spam");
        countWordFreqOfEachFile(hamTrainPath, "ham");
        
        filesSpam.addAll(wordFreqOfEachFileSpam.keySet());
        filesHam.addAll(wordFreqOfEachFileHam.keySet());
        filesSpamHam.addAll(filesSpam);
        filesSpamHam.addAll(filesHam);
        
        if (toFilterStopwords) {
            filterStopwords(stopWordsPath);
        }
        
        calWeights();
        //System.out.println(weightsOfEachWord);
    }
    
    /**
     * 
     * @param file
     * @return 1 if the classification result is spam, return 0 otherwise
     */
    public int LRClassification(File file) {
        // scan test file
        LinkedHashMap<String, Integer> wordFreqOfCurrFile = new LinkedHashMap<>();
        Scanner sc = null;
        try {
            sc = new Scanner(file);
            //sc.useDelimiter("[^\\w]");  // scan only words
            sc.useDelimiter("[^a-zA-Z]+");
            String nextWord;
            while (sc.hasNext()) {
                nextWord = sc.next();
                if (wordFreqOfCurrFile.containsKey(nextWord)) {
                    wordFreqOfCurrFile.put(nextWord, wordFreqOfCurrFile.get(nextWord) + 1);
                } else {
                    wordFreqOfCurrFile.put(nextWord, 1);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (sc != null) {
                sc.close();
            }
        }
        double classificationRule = w0;
        for (Entry<String, Integer> entry : wordFreqOfCurrFile.entrySet()) {
            if (weightsOfEachWord.containsKey(entry.getKey())) {
                classificationRule += weightsOfEachWord.get( entry.getKey() ) * entry.getValue();
            } 
        }
        //System.out.println(classificationRule); ///!!!
        if (classificationRule > 0) {
            return 1;
        } else {
            return 0;
        }
    }
    
}

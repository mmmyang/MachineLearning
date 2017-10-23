package NB;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Scanner;
import java.util.Set;

public class NaiveBayesMultinomial {
    private LinkedHashMap<String, Integer> wordFreqSpam = new LinkedHashMap<>(200);
    private LinkedHashMap<String, Integer> wordFreqHam = new LinkedHashMap<>(200);
    
    private LinkedHashMap<String, Double> wordProbSpam = new LinkedHashMap<>(200);   // probability is in log-scale
    private LinkedHashMap<String, Double> wordProbHam = new LinkedHashMap<>(200);    // probability is in log-scale
    
    private Set<String> vocabulary = new HashSet<>();
    private Set<String> stopwords = new HashSet<>();
    
    private int spamWordTotalNum = 0;
    private int hamWordTotalNum = 0;
    private int wordNumInVocabulary = 0;
    
    private int spamFileNum = 0;
    private int hamFileNum = 0;
    
    private double probSpam = 0.0;
    private double probHam = 0.0;
    
    /**
     * count the word frequency of each word in the data files
     * @param dataPath
     * @param dataClass is "spam" if the data is spam training data, and is "ham" if the data is ham training data
     */
    public void countWordFreq(String dataPath, String dataClass) {
        File dataFile = new File(dataPath);
        
        LinkedHashMap<String, Integer> wordFreq = null;
        if (dataClass.equalsIgnoreCase("spam")) {
            wordFreq = wordFreqSpam;
            spamFileNum = dataFile.listFiles().length;
        } else {
            wordFreq = wordFreqHam;
            hamFileNum = dataFile.listFiles().length;
        }
        
        for(File file: dataFile.listFiles()) {
            Scanner sc = null;
            try {
                sc = new Scanner(file);
                //sc.useDelimiter("[^\\w]");  // scan only words
                sc.useDelimiter("[^a-zA-Z]+");
                String nextWord;
                while (sc.hasNext()) {
                    nextWord = sc.next();
                    if (wordFreq.containsKey(nextWord)) {
                        wordFreq.put(nextWord, wordFreq.get(nextWord) + 1);
                    } else {
                        wordFreq.put(nextWord, 1);
                    }
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } finally {
                if (sc != null) {
                    sc.close();
                }
            }
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
            wordFreqSpam.remove(stopword);
            wordFreqHam.remove(stopword);
        }      
    }
    
    public void calWordProbGivenClass() {
        // count how many words in total there are in the training spam files
        int count = 0;
        for (Integer c : wordFreqSpam.values()) {
            count += c;
        }
        spamWordTotalNum = count;
        
        // count how many words in total there are in the training ham files
        count = 0;
        for (Integer c : wordFreqHam.values()) {
            count += c;
        }
        hamWordTotalNum = count;
        
        // union the spam and ham word set to get total vocabulary
        vocabulary.addAll(wordFreqSpam.keySet());
        vocabulary.addAll(wordFreqHam.keySet());
        wordNumInVocabulary = vocabulary.size();
        
        // calculate the probability of each word given class
        double prob = 0;
        double probLog = 0;
        for (String s : vocabulary) {
            // given class spam
            if (wordFreqSpam.containsKey(s)) {
                prob = ( wordFreqSpam.get(s) + 1.0 ) / ( spamWordTotalNum + wordNumInVocabulary );
            } else {
                prob = ( 0 + 1.0 ) / ( spamWordTotalNum + wordNumInVocabulary );
            }
            probLog = Math.log(prob);
            wordProbSpam.put(s, probLog);
            
            // given class ham
            if (wordFreqHam.containsKey(s)) {
                prob = ( wordFreqHam.get(s) + 1.0 ) / ( hamWordTotalNum + wordNumInVocabulary );
            } else {
                prob = ( 0 + 1.0 ) / ( hamWordTotalNum + wordNumInVocabulary );
            }
            probLog = Math.log(prob);
            wordProbHam.put(s, probLog);
        }
    }
    
    public void calPriorProbs() {
        double priorProb = 1.0 * spamFileNum / ( spamFileNum + hamFileNum );
        probSpam = Math.log(priorProb);
        priorProb = 1 - priorProb;
        probHam = Math.log(priorProb);
    }
    
    public void NBTraining(String spamTrainPath, String hamTrainPath, String stopWordsPath, boolean toFilterStopwords) {
        countWordFreq(spamTrainPath, "spam");
        countWordFreq(hamTrainPath, "ham");
        
        if (toFilterStopwords) {
            filterStopwords(stopWordsPath);
        }
        
        calWordProbGivenClass();
        calPriorProbs();
        //System.out.println(vocabulary);
        //System.out.println(wordFreqHam);
        //System.out.println(wordProbHam);
        //System.out.println(spamWordTotalNum);
    }
    
    /**
     * 
     * @param file is the file to be classified
     * @return 1 if the classification result is spam, return 0 otherwise
     */
    public int NBClassification(File file) {
        double currProbOfSpam = 0.0;
        double currProbOfHam = 0.0;
        
        Scanner sc = null;
        try {
            sc = new Scanner(file);
            sc.useDelimiter("[^\\w]");  // scan only words
            String nextWord;
            while (sc.hasNext()) {
                nextWord = sc.next();
                if (wordProbSpam.containsKey(nextWord)) {
                    currProbOfSpam += wordProbSpam.get(nextWord);
                }
                if (wordProbHam.containsKey(nextWord)) {
                    currProbOfHam += wordProbHam.get(nextWord);
                }
            }  
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } finally {
            if (sc != null) {
                sc.close();
            }
        }
        currProbOfSpam += probSpam;
        currProbOfHam += probHam;
        
        //System.out.println("< " +currProbOfSpam + " " +currProbOfHam +" >");
        
        if (currProbOfSpam > currProbOfHam) {
            return 1;   // the result is spam
        } else {
            return 0;   // the result is ham (not spam)
        }
    }
}

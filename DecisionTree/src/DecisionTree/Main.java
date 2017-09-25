package DecisionTree;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * this class contains the main method to run the program
 * @author FenglanYang
 *
 */
public class Main {
    
    public static void main(String[] args) throws MyException {
        // parse arguments
        if (args.length != 6) {
            throw new MyException("Wrong Input! Please enter the arguments as: <L> <K> <training-set> <validation-set> <test-set> <to-print>");
        }
        int L = Integer.parseInt(args[0]);
        int K = Integer.parseInt(args[1]);
        ArrayList<ArrayList<AttrData>> trainingSet = csvParser(args[2]);
        ArrayList<ArrayList<AttrData>> validationSet = csvParser(args[3]);
        ArrayList<ArrayList<AttrData>> testSet = csvParser(args[4]);
        boolean toPrint = args[5].equalsIgnoreCase("yes");
        
        DecisionTree decisionTree = new DecisionTree();
        decisionTree.buildDecisionTree(trainingSet, decisionTree.root, "entropy");
        System.out.println("Accuracy for decision tree constructed using entropy heuristic is: " + decisionTree.getAccuracy(testSet, decisionTree.root));
        if (toPrint) {
            System.out.println("The decision tree is:");
            System.out.println(decisionTree);
        }
        
        decisionTree.postPruning(validationSet, L, K);
        System.out.println("Accuracy for decision tree after post pruning is: " + decisionTree.getAccuracy(testSet, decisionTree.root));
        if (toPrint) {
            System.out.println("The decision tree after post pruning is:");
            System.out.println(decisionTree);
        }
        
        decisionTree = new DecisionTree();
        decisionTree.buildDecisionTree(trainingSet, decisionTree.root, "varImpurity");
        System.out.println("Accuracy for decision tree constructed using variance impurity heuristic is: " + decisionTree.getAccuracy(testSet, decisionTree.root));
        if (toPrint) {
            System.out.println("The decision tree is:");
            System.out.println(decisionTree);
        }
        
        decisionTree.postPruning(validationSet, L, K);
        System.out.println("Accuracy for decision tree after post pruning is: " + decisionTree.getAccuracy(testSet, decisionTree.root));
        if (toPrint) {
            System.out.println("The decision tree after post pruning is:");
            System.out.println(decisionTree);
        }
    }
    
    /**
     * Parse a CSV file for data set.
     * Return the data set as a ArrayList<ArrayList<AttrData>>.
     * @param csvPath
     * @return
     */
    private static ArrayList<ArrayList<AttrData>> csvParser(String csvPath) {
        ArrayList<ArrayList<AttrData>> dataSet = new ArrayList<>();
        
        try {
            Scanner sc = new Scanner(new File(csvPath));
            sc.useDelimiter(",");
            
            String[] attrNames = sc.nextLine().split(",");  // stores the attribute names
            int attrNum = attrNames.length;
            int i;
            while (sc.hasNextLine()) {
                String[] attrValues = sc.nextLine().split(",");
                ArrayList<AttrData> tupleData = new ArrayList<>();
                for (i = 0; i < attrNum; i++) {
                    tupleData.add(new AttrData(attrNames[i], Integer.parseInt(attrValues[i])));
                }
                dataSet.add(tupleData);
            }
            
            sc.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();                                  
        }
        return dataSet;
    }
}

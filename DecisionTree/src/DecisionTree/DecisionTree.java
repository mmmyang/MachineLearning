package DecisionTree;

import java.util.ArrayList;
import java.util.Random;

/**
 * Implemented the decision tree algorithm and the post pruning algorithm.
 * Contains method to calculate the decision tree accuracy.
 * Contains method to print the decision tree.
 * @author FenglanYang
 *
 */
public class DecisionTree {
    public DTNode root = new DTNode();
    public int nonLeafNodeNum = 0;
    
    /**
     * Build a decision tree using the provided training set and heuristic.
     * @param dataSet is the training data
     * @param currNode is the root of the subtree
     * @param heuristic
     * @throws MyException 
     */
    public void buildDecisionTree(ArrayList<ArrayList<AttrData>> dataSet, DTNode currNode, String heuristic) throws MyException {
        Heuristics heur = new Heuristics();
        double maxGain = 0;
        String bestAttr = null;
        int bestAttrIndex = -1;
        DTNode child0;  // left child of the currNode
        DTNode child1;  // right child of the currNode
        ArrayList<ArrayList<AttrData>> bestSubSet0 = new ArrayList<>(); // data set for child0; best attribute value = 0
        ArrayList<ArrayList<AttrData>> bestSubSet1 = new ArrayList<>(); // data set for child1; best attribute value = 1
        
        // Compute the entropy/impurity of the current node's data set
        if (heuristic.equalsIgnoreCase("entropy")) {
            currNode.entropy = heur.getEntropy(dataSet);
        } else if (heuristic.equalsIgnoreCase("varImpurity")) {
            currNode.entropy = heur.getVarImpurity(dataSet);
        } else {
            throw new MyException("The heuristic is not acceptable. Please enter the heuristic as \"entropy\" or \"varImpurity\".");
        }
        
        int tupleNum = dataSet.size();  // number of training examples
        int AttrNum = dataSet.get(0).size();    // number of attributes including the class attribute
        ArrayList<ArrayList<AttrData>> subSet0; // store sub-dataSet with attribute value = 0
        ArrayList<ArrayList<AttrData>> subSet1; // store sub-dataSet with attribute value = 1
        ArrayList<AttrData> aTuple; // store a training example
        ArrayList<Integer> subSetTupleNums;
        ArrayList<Double> subSetEntropies;
        double currGain;
        
        // traverse all the attributes except the class attribute to find the best attribute at this node
        for (int i = 0; i < AttrNum - 1; i++) {
            subSet0 = new ArrayList<>();
            subSet1 = new ArrayList<>();
            // traverse all training examples to split data set according to current attribute value
            for (int j = 0; j < tupleNum; j++) {
                aTuple = new ArrayList<>();
                aTuple.addAll(dataSet.get(j));  //
                if (dataSet.get(j).get(i).value == 0) {
                    subSet0.add(aTuple);
                } else {
                    subSet1.add(aTuple);
                }
            }
            
            subSetTupleNums = new ArrayList<>();
            subSetTupleNums.add(subSet0.size());
            subSetTupleNums.add(subSet1.size());
            
            // calculate entropies of split subsets.
            subSetEntropies = new ArrayList<>();
            if (heuristic.equalsIgnoreCase("entropy")) {
                subSetEntropies.add(heur.getEntropy(subSet0));
                subSetEntropies.add(heur.getEntropy(subSet1));
            } else if (heuristic.equalsIgnoreCase("varImpurity")) {
                subSetEntropies.add(heur.getVarImpurity(subSet0));
                subSetEntropies.add(heur.getVarImpurity(subSet1));
            } else {
                throw new MyException("The heuristic is not acceptable. Please enter the heuristic as \"entropy\" or \"varImpurity\".");
            }
            
            // calculate the gain according to current data split
            currGain = heur.getGain(currNode.entropy, tupleNum, subSetTupleNums, subSetEntropies); 
            if (currGain > maxGain) {   // this is a better attribute
                maxGain = currGain;
                bestAttr = dataSet.get(0).get(i).name;
                bestAttrIndex = i;
                bestSubSet0 = subSet0;  //
                bestSubSet1 = subSet1;  //
            }
        }
        
        if (bestAttr != null) { // found the best attribute to split data set
            currNode.bestAttr = bestAttr;
            currNode.trainingData = dataSet;
            currNode.index = ++nonLeafNodeNum;
            
            child0 = new DTNode();
            child1 = new DTNode();
            
            // remove best attribute for current node in the two sub-dataSet
            // to avoid choosing same attribute in descendants
            for (ArrayList<AttrData> tuple : bestSubSet0) {
                tuple.remove(bestAttrIndex);
            }
            for (ArrayList<AttrData> tuple : bestSubSet1) {
                tuple.remove(bestAttrIndex);
            }
            
            //child0.trainingData = bestSubSet0;
            //child1.trainingData = bestSubSet1;
            
            currNode.children = new ArrayList<DTNode>();
            currNode.children.add(child0);
            currNode.children.add(child1);
            
            // recursively find best attribute to split data and build subtrees
            buildDecisionTree(bestSubSet0, child0, heuristic);
            buildDecisionTree(bestSubSet1, child1, heuristic);
        } else {    // this is the leaf node since data set is pure for all attributes
            currNode.trainingData = dataSet;
            currNode.classVal = dataSet.get(0).get(AttrNum - 1).value;
        }
    }
    
    /**
     * Post pruning method used to avoid decision tree overfitting.
     * @param dataSet is the validation set used to prune the tree
     * @param L
     * @param K
     */
    public void postPruning(ArrayList<ArrayList<AttrData>> dataSet, int L, int K) {
        // !!! need implementation
        DTNode bestTreeRoot = root;
        DTNode newTreeRoot;
        int oriNum = nonLeafNodeNum;
        int bestNum = nonLeafNodeNum;
        int i, j, N, P;
        Random random = new Random();
        
        // pruning the tree
        for (i = 0; i < L; i++) {
            newTreeRoot = new DTNode(root); // clone original decision tree to a new tree
            
            N = oriNum;
            for(j = 0; j < random.nextInt(K) && N > 0; j++) {   // if N == 0 (root is leaf), then no need to further prune
                // replace the subtree rooted at a node with the index by a leaf node
                P = random.nextInt(N);
                replaceByLeafNode(newTreeRoot, P);
                
                // order non-leaf nodes in pre-order and assign N non-leaf node number     
                nonLeafNodeNum = 0;
                orderNonLeafNodes(newTreeRoot);
                N = nonLeafNodeNum;
            }
            
            if (getAccuracy(dataSet, newTreeRoot) > getAccuracy(dataSet, bestTreeRoot)) {
                bestTreeRoot = newTreeRoot;
                bestNum = nonLeafNodeNum;
            }
        }
        root = bestTreeRoot;
        nonLeafNodeNum = bestNum;
    }
    
    /**
     * Replace the subtree rooted at a node with the index by a leaf node.
     * Assign the majority class of training data at the index to the leaf node.
     * @param root is the root of the tree to be pruned
     * @param index
     */
    private void replaceByLeafNode(DTNode root, int index) {
        if (root.index == index) {
            // count the example numbers of class value 0 and 1
            int[] classCount = new int[2];
            for (ArrayList<AttrData> tuple : root.trainingData) {
                classCount[tuple.get(tuple.size() - 1).value]++;
            }
            
            // replace the subtree by a leaf node
            root.classVal = classCount[0] > classCount[1] ? 0 : 1;
            root.children = null;
            root.bestAttr = null;
            root.index = -1;
        } else if (root.children != null) { // otherwise its a leaf node, just return
            replaceByLeafNode(root.children.get(0), index);
            replaceByLeafNode(root.children.get(1), index);
        }
    }
    
    /**
     * Recursively order the tree in pre-order.
     * @param root is the root of the tree to be ordered
     */
    private void orderNonLeafNodes(DTNode root) {
        if (root != null && root.children != null) {    // non-leaf node
            root.index = ++nonLeafNodeNum;
            orderNonLeafNodes(root.children.get(0));
            orderNonLeafNodes(root.children.get(1));
        }
    }
    
    /**
     * Evaluate the accuracy of decision tree rooted at the passed root on the passed validation dataSet.
     * Accuracy = percentage of correctly classified examples.
     * @param dataSet is the validation set
     * @param root
     * @return
     */
    public double getAccuracy(ArrayList<ArrayList<AttrData>> dataSet, DTNode root) {
        double accuracy = 0;
        
        // traverse the validation data set and check how many example are predicted correctly
        for (ArrayList<AttrData> tuple : dataSet) {
            if (isCorrectPrediction(tuple, root)) { // decision tree predict correctly for this example
                accuracy++;
            }
        }
        
        return accuracy / dataSet.size() * 100;
    }
    
    /**
     * Check whether the decision tree can correctly predict the class result of the passed dataTuple.
     * @param dataTuple
     * @param root
     * @return
     */
    private boolean isCorrectPrediction(ArrayList<AttrData> dataTuple, DTNode root) {
        if (root.children == null) {    // a leaf node, check the class value
            return root.classVal == dataTuple.get(dataTuple.size() - 1).value;
        } else {    // go the check the next level
            int leftOrRight = 0;
            for (AttrData attrData : dataTuple) {
                if (attrData.name.equals(root.bestAttr)) {
                    leftOrRight = attrData.value;
                }
            }
            return isCorrectPrediction(dataTuple, root.children.get(leftOrRight));
        }
    }
    
    public String toString() {
        return printDecisionTree(root, 0);
    }
    
    /**
     * Return a String representation of the subtree.
     * The tree representation format is:
     * wesley = 0 :
     * | honor = 0 : 
     * | | barclay = 0 : 1 
     * | | barclay = 1 : 0 
     * | honor = 1 : 
     * | | tea = 0 : 0 
     * | | tea = 1 : 1 
     * wesley = 1 : 0 
     * @param currNode is root of the subtree
     * @param level is the level of currNode in the tree
     * @return
     */
    private String printDecisionTree(DTNode currNode, int level) {
        StringBuilder sBuilder = new StringBuilder();
        
        if (currNode.children == null) {    // a leaf node
            sBuilder.append(" " + currNode.classVal + "\n");
        } else {    // non-leaf node
            for (int i = 0; i < currNode.children.size(); i++) {    // traverse all children
                for (int j = 0; j < level; j++) {
                    sBuilder.append("| ");
                }
                sBuilder.append(currNode.bestAttr + " = " + i + " :");
                if (currNode.children.get(i).children == null) {    // this child of currNode is a leaf
                    sBuilder.append(" " + currNode.children.get(i).classVal + "\n");
                } else {
                    sBuilder.append("\n" + printDecisionTree(currNode.children.get(i), level + 1));
                }
            }
        }
        
        return sBuilder.toString();
    }
}
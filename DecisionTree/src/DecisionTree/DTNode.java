package DecisionTree;

import java.util.ArrayList;

/**
 * A node in the decision tree.
 * @author FenglanYang
 *
 */
public class DTNode {
    public ArrayList<DTNode> children;  // list of child nodes
    public String bestAttr; // the best attribute chosen to split data set at this node; only for non-leaf nodes
    public Double entropy;  // entropy (represents impurity) at this node
    public ArrayList<ArrayList<AttrData>> trainingData;    // training data set at this node
    public int index;   // index for post pruning; 1 - N for non-leaf nodes in pre-order
    public int classVal;    // class value at this node; only for leaf nodes
    
    public DTNode() {
        children = null;
        bestAttr = null;
        entropy = 0.0;
        trainingData = null;    //
        index = -1;
        classVal = -1;  //
    }
    
    public DTNode(ArrayList<DTNode> children, String bestAttr, Double entropy, 
            ArrayList<ArrayList<AttrData>> traningData, int index, int classVal) {
        
        this.bestAttr = "" + bestAttr;  // create a new String object
        this.entropy = entropy;
        this.index = index;
        this.classVal = classVal;
        
        // set the trainingData
        this.trainingData = new ArrayList<>();
        for (ArrayList<AttrData> tupleData : trainingData) {
            ArrayList<AttrData> tupleDataCopy = new ArrayList<>();
            for (AttrData attrData : tupleData) {
                tupleDataCopy.add(new AttrData(attrData.name, attrData.value));
            }
            this.trainingData.add(tupleDataCopy);
        }
        
        // set the children
        if (children == null) {
            this.children = null;
        } else {
            this.children = new ArrayList<DTNode>();
            for (DTNode child : children) {
                this.children.add(new DTNode(child));
            }
        }
    }
    
    /**
     * Copy constructor used to clone a DTNode
     * @param node is the node to be cloned
     */
    public DTNode(DTNode node) {
        this(node.children, node.bestAttr, node.entropy, node.trainingData, node.index, node.classVal);
    }
}

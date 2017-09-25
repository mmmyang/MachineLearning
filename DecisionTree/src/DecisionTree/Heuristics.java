package DecisionTree;

import java.util.ArrayList;

/**
 * Contains methods calculating heuristics for selecting best Attributes.
 * @author FenglanYang
 *
 */
public class Heuristics {
    private static int classValNum = 2;
    
    /**
     * Calculate Entropy of the dataSet.
     * Entropy = -sumOf(Ki/K*log(Ki/K)).
     * @param dataSet
     * @return
     */
    public double getEntropy(ArrayList<ArrayList<AttrData>> dataSet) {
        double[] classValCounts = new double[classValNum];
        for (ArrayList<AttrData> tupleData : dataSet) {
            classValCounts[tupleData.get(tupleData.size() - 1).value]++;
        }
        double tupleNum = dataSet.size();
        double entropy = 0.0;
        for (double count : classValCounts) {
            if (count != 0) {
                entropy -= count * (Math.log(count)/Math.log(2));
            }
        }
        entropy = entropy / tupleNum + (Math.log(tupleNum)/Math.log(2)); 
        return entropy;
    }
    
    /**
     * Calculate Variance Impurity of the dataSet.
     * Variance Impurity = sumOf(Ki/K).
     * @param dataSet
     * @return
     */
    public double getVarImpurity(ArrayList<ArrayList<AttrData>> dataSet) {
        double[] classValCounts = new double[classValNum];
        for (ArrayList<AttrData> tupleData : dataSet) {
            classValCounts[tupleData.get(tupleData.size() - 1).value]++;
        }
        double tupleNum = dataSet.size();
        double valImpurity = 1.0;
        for (double count : classValCounts) {
            valImpurity *= count / tupleNum;
        }
        return valImpurity;
    }
    
    /**
     * Calculate the gain with respect to different heuristics.
     * @param currEntropy
     * @param currTupleNum
     * @param subSetTupleNums
     * @param subSetEntropies
     * @return
     */
    public double getGain(double currEntropy, double currTupleNum, ArrayList<Integer> subSetTupleNums, ArrayList<Double> subSetEntropies) {
        double gain = currEntropy;
        for (int i = 0; i < subSetTupleNums.size(); i++) {
            gain -= subSetTupleNums.get(i) / currTupleNum * subSetEntropies.get(i);
        }
        return gain;
    }
}

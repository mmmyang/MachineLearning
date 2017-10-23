package kMeans;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import javax.imageio.ImageIO;

/**
 * 
 * @author FenglanYang
 *
 */
public class KMeans {
    
    /**
     * 
     * @param args three arguments: <input-image-location> <k> <output-image-location>
     */
    public static void main(String [] args){
        if (args.length < 3){
            System.out.println("Usage: Kmeans <input-image> <k> <output-image>");
            return;
        }
        try {
            BufferedImage originalImage = ImageIO.read(new File(args[0]));
            int k=Integer.parseInt(args[1]);
            System.out.println("Start calculating...");
            BufferedImage kmeansJpg = kmeans_helper(originalImage,k);
            ImageIO.write(kmeansJpg, "jpg", new File(args[2]));
            System.out.println("Calculate ended...");
        } catch(IOException e) {
            System.out.println(e.getMessage());
        }   
    }
    
    /**
     * get the compressed image using KMean algorithm
     * @param originalImage
     * @param k
     * @return
     */
    private static BufferedImage kmeans_helper(BufferedImage originalImage, int k) {
        // copy the originalImage to the kmeansImage
        int w=originalImage.getWidth();
        int h=originalImage.getHeight();
        BufferedImage kmeansImage = new BufferedImage(w,h,originalImage.getType());
        Graphics2D g = kmeansImage.createGraphics();
        g.drawImage(originalImage, 0, 0, w,h , null);
        // Read rgb values from the image
        int[] rgb=new int[w*h];
        int count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                rgb[count++]=kmeansImage.getRGB(i,j);
            }
        }
        
        // Call kmeans algorithm: update the rgb values
        kmeans(rgb,k);
        
        // Write the new rgb values to the image
        count=0;
        for(int i=0;i<w;i++){
            for(int j=0;j<h;j++){
                kmeansImage.setRGB(i,j,rgb[count++]);
            }
        }
        return kmeansImage;
    }

    // Update the array rgb by assigning each entry in the rgb array to its cluster center
    private static void kmeans(int[] rgb, int k) {
        int[] currClusterCenters = new int[k]; 
        int[] newMeanRed = new int[k];
        int[] newMeanGreen = new int[k];
        int[] newMeanBlue = new int[k];
        int[] pointsNumAssignToClutser = new int[k];
        
        int[] assignments = new int[rgb.length];    // cluster assignments for each point
        double[] ColorDistWithCluster = new double[k];
        
        Random random = new Random();
        // Pick K random points as cluster centers (means)

        int x;
        for(int i=0; i < k; i++) {
            x = rgb[random.nextInt(rgb.length)]; // pick k colors from the w*h points in the figure as the cluster center
            currClusterCenters[i] = x;
        }
        
        int changedMeanNum = k;
        int iterate = 1;
        while (changedMeanNum > 0) {    //otherwise, if no mean is updated, stop
            System.out.println("iterate " + iterate++);
            
            for(int i=0; i < k; i++) {
                newMeanRed[i] = 0;
                newMeanGreen[i] = 0;
                newMeanBlue[i] = 0;
                pointsNumAssignToClutser[i] = 0;
            }
           
            // for each point in the figure, update assignments
            for (int i = 0; i < rgb.length; i++) {
                assignments[i] = 0;
                double minDist = Double.MAX_VALUE;
                Color currPointColor = new Color(rgb[i]);
                for (int j = 0; j < k; j++) {   // for each point in the mean set (clusters)
                    Color currClusterColor = new Color(currClusterCenters[j]);
                    ColorDistWithCluster[j] = calColorDist(currPointColor, currClusterColor);
                    if (ColorDistWithCluster[j] < minDist) {
                        minDist = ColorDistWithCluster[j];
                        assignments[i] = j; // assign point i to cluster j (currently closest cluster)
                    }
                }
                
                int currAssign = assignments[i];
                newMeanRed[currAssign] += currPointColor.getRed();    // add the assigned point color to the corresponding cluster for later average calculation to update mean
                newMeanGreen[currAssign] += currPointColor.getGreen();
                newMeanBlue[currAssign] += currPointColor.getBlue();
                pointsNumAssignToClutser[currAssign] += 1;  // count how many points are assigned to each cluster
            }
            
            // for each cluster in the mean set, update mean
            changedMeanNum =0;
            int newMeanR;
            int newMeanG;
            int newMeanB;
            int newMean;
            for (int j = 0; j < k; j++) {
                if (pointsNumAssignToClutser[j] != 0) {
                    newMeanR = newMeanRed[j] / pointsNumAssignToClutser[j];
                    newMeanG = newMeanGreen[j] / pointsNumAssignToClutser[j];
                    newMeanB = newMeanBlue[j] / pointsNumAssignToClutser[j];
                    newMean = (new Color(newMeanR, newMeanG, newMeanB)).getRGB();
                    if (newMean != currClusterCenters[j]) {
                        currClusterCenters[j] = newMean;    // update cluster center to new mean
                        changedMeanNum ++;
                    }     
                }
            }
        }   // end of update
        
        // update the array rgb
        int newColor;
        for(int i = 0; i < rgb.length; i++) {
            newColor = currClusterCenters[assignments[i]];
            rgb[i] = newColor;
        }
        
    }
    
    /**
     * calculate the color difference of the two specific points
     * @param point1
     * @param point2
     * @return
     */
    private static  double calColorDist(Color point1, Color point2) {
        double Rdist = Math.abs(point1.getRed() - point2.getRed());
        double Gdist = Math.abs(point1.getGreen() - point2.getGreen());
        double Bdist = Math.abs(point1.getBlue() - point2.getBlue());
        return Math.sqrt(Rdist*Rdist + Gdist*Gdist + Bdist*Bdist);
    }

}

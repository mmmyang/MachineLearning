package recommendation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

public class MovieRecommendation {
    private static JavaSparkContext sc;
    
    @SuppressWarnings("serial")
    public static void main(String[] args) {
        Logger.getLogger("org").setLevel(Level.ERROR);
        Logger.getLogger("akka").setLevel(Level.ERROR);
        
        // parse input arguments
        System.out.println("parsing input arguments...");
        if (args.length != 3) {
            System.out.println("Wrong Input! Please enter the arguments as: <data-file-location> <userToRecommend> <Recommendation-Num>");
            System.exit(1);
        }
        
        SparkConf conf = new SparkConf().setAppName("Movie Recommendation with ALS").setMaster("local[*]");
        sc = new JavaSparkContext(conf);
        
        
        // load data
        System.out.println("loading data...");
        String ratingFileLoc = args[0] + "/" + "ratings.dat";
        String movieFileLoc = args[0] + "/" + "movies.dat";
        int userToRecommend = Integer.parseInt(args[1]);
        int recommendationNum = Integer.parseInt(args[2]);
        // rating file has the format: UserID::MovieID::Rating::Timestamp
        final JavaRDD<String> ratingData = sc.textFile(ratingFileLoc);
        // movie file has the format: MovieID::Title::Genres
        JavaRDD<String> movieData = sc.textFile(movieFileLoc);
        
        System.out.println("parsing data...");
        // parse the rating data to <Integer, Rating> pairs. Each Rating is a <user, movie, rating> data
        JavaRDD<Tuple2<Integer, Rating>> ratings = ratingData.map(  // Return a new RDD by applying a function to all elements of this RDD
                new Function<String, Tuple2<Integer, Rating>>() {   // convert String to Tuple2<Integer, Rating>
                    public Tuple2<Integer, Rating> call(String s) throws Exception {
                        String[] currRow = s.split("::");
                        Integer currKey = Integer.parseInt(currRow[3]) % 10; // generate a key using the last digit of Timestamp: for use of later data set splitting
                        Rating rating = new Rating(Integer.parseInt(currRow[0]), Integer.parseInt(currRow[1]), Double.parseDouble(currRow[2])); // create a new Rating using current row information
                        return new Tuple2<Integer, Rating>(currKey, rating);
                    }
                }
        );
        
        // parse the movie data to <Integer, String> pairs. The Integer is a movie id and the String is the movie title
        Map<Integer, String> movies = movieData.mapToPair(    // Return a new RDD by applying a function to all elements of this RDD
                new PairFunction<String, Integer, String>() {   // convert String to a <Integer, String> pair
                    public Tuple2<Integer, String> call(String s) throws Exception {
                        String[] currRow = s.split("::");
                        return new Tuple2<Integer, String>(Integer.parseInt(currRow[0]), currRow[1]);
                    }
                }
        ).collectAsMap();   // Return the key-value pairs as a Map
        
        // print the data information
        long ratingNum = ratings.count();   // count the number of ratings
        long movieNum = ratings.map(    // count the number of movies
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().product();    // _1() is the Integer key and _2() is the Rating object
                    }
                }
        ).distinct().count();
        long userNum = ratings.map(   // count the number of users
                new Function<Tuple2<Integer, Rating>, Object>() {
                    public Object call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user();
                    }
                }
        ).distinct().count();
        System.out.println("The data contains " + ratingNum + " ratings from " + userNum + " users on " + movieNum + " movies.");
        
        
        // split the data to three non-overlapping subsets: training, validation, and test data
        System.out.println("splitting data set...");
        int numPartitions = 10;
        // get training data set
        JavaRDD<Rating> training = ratings.filter(  // filter the data set with the predicate
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() < 6;  // assign all data with key<6 to training data
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();  // get all the tuples satisfying the filter condition
                    }
                }
        ).repartition(numPartitions).cache();   // re-split the training data to 10 partitions and persist the data in memory

        StorageLevel storageLevel = new StorageLevel();
        // get validation data set
        JavaRDD<Rating> validation = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 6 && tuple._1() < 8;   // assign all data with 6<=key<8 to training data
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();  // get all the tuples satisfying the filter condition
                    }
                }
        ).repartition(numPartitions).persist(storageLevel); // re-split the training data to 10 partitions

        // get test data set
        JavaRDD<Rating> test = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._1() >= 8; // assign all data with key>=8 to training data
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();  // get all the tuples satisfying the filter condition
                    }
                }
        ).persist(storageLevel);    

        long trainingNum = training.count();
        long validationNum = validation.count();
        long testNum = test.count();
        
        System.out.println("The whole data set is split into three non-overlapping subsets: ");
        System.out.println("Training data number: " + trainingNum);
        System.out.println("Validation data number: " + validationNum);
        System.out.println("Test data number: " + testNum);
        
        
        // train multiple models
        System.out.println("training multiple models using different combinations of rank, regularization constant, and iteration number...");
        int[] ranks = {18};  // matrix rank
        float[] lambdas = {0.09f};    // regularization constant
        int[] numIters = {25};  // number of iterations

        double bestvalidationRMSE = Double.MAX_VALUE;
        int bestRank = 0;
        float bestLambda = -1.0f;
        int bestNumIter = -1;
        MatrixFactorizationModel bestModel = null;
        
        // train models using different combinations of rank, regularization constant, and iteration number
        // and assign the model with the smallest RMSE on the validation set as the best model
        for (int currRank : ranks) {
            for (float currLambda : lambdas) {
                for (int currNumIter : numIters) {
                    // train the model representing the result of matrix factorization
                    MatrixFactorizationModel currModel = ALS.train(JavaRDD.toRDD(training), currRank, currNumIter, currLambda);
                    // calculate the RMSE of this model on validation data set
                    Double validationRMSE = calRMSE(currModel, validation);
                    System.out.println("RMSE on validation set = " + validationRMSE + " for the model trained with rank = " + currRank + ", lambda = " + currLambda + ", and numIter = " + currNumIter + ".");
                    
                    // update the best model
                    if (validationRMSE < bestvalidationRMSE) {
                        bestModel = currModel;
                        bestvalidationRMSE = validationRMSE;
                        bestRank = currRank;
                        bestLambda = currLambda;
                        bestNumIter = currNumIter;
                    }
                }
            }
        }

        // calculate and output the RMSE of the best model on the test data set
        Double trainingRMSE = calRMSE(bestModel, training);
        Double testRMSE = calRMSE(bestModel, test);
        System.out.println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda + ", and numIter = " + bestNumIter + ".");
        System.out.println("The RMSE of the best model on the training set is " + trainingRMSE + ".");
        System.out.println("The RMSE of the best model on the validation set is " + bestvalidationRMSE + ".");
        System.out.println("The RMSE of the best model on the test set is " + testRMSE + ".");
        
        /*
        // saving model as local file
        System.out.println("saving model...");
        bestModel.save(sc.sc(), "data/trainedModel");
        */
        
        /*
        System.out.println("loading model...");
        bestModel = MatrixFactorizationModel.load(sc.sc(), "data/trainedModel");
        */
        
        // get Recommendations
        List<Rating> recommendations = getRecommendations(userToRecommend, bestModel, ratings, movies, recommendationNum);
        System.out.println("The recommended movies for user " + userToRecommend + " are: ");
        int i = 0;
        for (Rating recommendation : recommendations) {
            Integer movieId = recommendation.product();
            if (movies.containsKey(movieId)) {
                System.out.println(++i + ".\t" + movies.get(movieId));
            }
        }
        
    }
    
    /**
     * calculate the RMSE (Root Mean Squared Error) of the model on the data
     * @param model
     * @param data
     * @return
     */
    @SuppressWarnings("serial")
    public static Double calRMSE(MatrixFactorizationModel model, JavaRDD<Rating> data) {
        // get the <user, product> map from the data
        JavaRDD<Tuple2<Object, Object>> userProducts = data.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        // get the prediction of ratings of all users for all products in the data using the model
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map( // predict the ratings of all users for all products
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                )
        );
        // convert the rating data to the same format as predictions
        JavaPairRDD<Tuple2<Integer, Integer>, Double> ratingsNewFormat = JavaPairRDD.fromJavaRDD(
                data.map(   // get the <Tuple2<Integer, Integer>, Double>> format of the data
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                return new Tuple2<Tuple2<Integer, Integer>, Double>(new Tuple2<Integer, Integer>(r.user(), r.product()), r.rating());
                            }
                        }
                )
        );
        // get the <rating, prediction> pairs of all <user, product> pairs
        JavaRDD<Tuple2<Double, Double>> ratingsPredictionsPairs = ratingsNewFormat.join(predictions).values();
        
        // calculate the mean square error
        double MSE =  JavaDoubleRDD.fromRDD(ratingsPredictionsPairs.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double error = pair._1() - pair._2();
                        return error * error;
                    }
                }).rdd()
        ).mean();

        return Math.sqrt(MSE);
    }
    
    /**
     * get the recommendation for the passed user using the passed model
     * @param userId
     * @param model
     * @param ratings
     * @param products
     * @param recommendationNum
     * @return
     */
    @SuppressWarnings("serial")
    private static List<Rating> getRecommendations(final int userId, MatrixFactorizationModel model, JavaRDD<Tuple2<Integer, Rating>> ratings, Map<Integer, String> products, int recommendationNum) {
        List<Rating> recommendations;

        // get the ratings of the passed user
        JavaRDD<Rating> userRatings = ratings.filter(
                new Function<Tuple2<Integer, Rating>, Boolean>() {  // get all <key, Rating> pairs of this user
                    public Boolean call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2().user() == userId;
                    }
                }
        ).map(
                new Function<Tuple2<Integer, Rating>, Rating>() {   // get all Ratings of this user
                    public Rating call(Tuple2<Integer, Rating> tuple) throws Exception {
                        return tuple._2();
                    }
                }
        );
        // get the movie ids of the movies the user rated
        JavaRDD<Tuple2<Object, Object>> userProducts = userRatings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        
        // get the whole movie id set
        List<Integer> productSet = new ArrayList<Integer>();
        productSet.addAll(products.keySet());

        // get the movie id set without movies rated by the user
        Iterator<Tuple2<Object, Object>> productIterator = userProducts.toLocalIterator();
        while(productIterator.hasNext()) {
            Integer productId = (Integer)productIterator.next()._2();
            if(productSet.contains(productId)){
                productSet.remove(productId);
            }
        }
        
        JavaRDD<Integer> candidates = sc.parallelize(productSet);   // all movies haven't been rated by the users
        
        // get the <user, movie> pairs where movies are movies haven't been rated by the users
        JavaRDD<Tuple2<Integer, Integer>> userCandidates = candidates.map(
                new Function<Integer, Tuple2<Integer, Integer>>() {
                    public Tuple2<Integer, Integer> call(Integer integer) throws Exception {
                        return new Tuple2<Integer, Integer>(userId, integer);
                    }
                }
        );

        // get recommendations for the passed user
        recommendations = model.predict(JavaPairRDD.fromJavaRDD(userCandidates)).collect();
        // make the recommendations list modifiable
        ArrayList<Rating> tempList = new ArrayList<Rating>();
        tempList.addAll(recommendations);
        recommendations = tempList;
        
        // sort the recommended products by the rating
        Collections.sort(recommendations, new Comparator<Rating>() {
            public int compare(Rating r1, Rating r2) {
                return r1.rating() < r2.rating() ? -1 : r1.rating() > r2.rating() ? 1 : 0;
            }
        });

        // get the top recommendationNum recommended movies from the recommendation list
        recommendations = recommendations.subList(0, recommendationNum);

        return recommendations;
    }
    
}

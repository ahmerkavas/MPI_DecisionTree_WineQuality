/*
 * MPI Decision Tree for the Wine Quality dataset
 *
 * This program trains and evaluates a small Decision Tree classifier using
 * distributed-memory parallelism with MS-MPI. Rank 0 reads the Wine Quality
 * white wine CSV file, converts raw quality scores into three classes, and
 * distributes uneven row partitions to all MPI ranks.
 *
 * Class mapping:
 *   quality <= 5  -> class 0, low
 *   quality == 6  -> class 1, medium
 *   quality >= 7  -> class 2, high
 *
 * MPI collectives used:
 *   MPI_Bcast    - share dataset metadata with all ranks
 *   MPI_Scatterv - distribute uneven feature and label partitions
 *   MPI_Gather   - report how many rows each process received
 *   MPI_Allreduce- combine distributed class counts and split statistics
 *   MPI_Reduce   - combine final accuracy counts and timing values on rank 0
 *   MPI_Barrier  - synchronize ranks before timing starts
 *
 * Goal:
 *   Demonstrate a meaningful MPI-parallel machine learning workflow where all
 *   ranks train from distributed data using collective communication. The goal
 *   is parallel and distributed computing practice, not maximum ML accuracy.
 */
/*
 * _CRT_SECURE_NO_WARNINGS suppresses MSVC warnings for standard C functions
 * such as fopen and strtok while keeping the code portable C-style.
 */
#define _CRT_SECURE_NO_WARNINGS

#include <ctype.h>
#include <float.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * FEATURE_COUNT is fixed by the Wine Quality dataset: the first 11 columns are
 * numeric input features and the last column is the quality target.
 */
#define FEATURE_COUNT 11

/*
 * CLASS_COUNT is 3 because raw wine quality scores are grouped into low,
 * medium, and high classes for classification.
 */
#define CLASS_COUNT 3

/*
 * NUM_THRESHOLDS controls how many candidate split thresholds each feature
 * tests at each tree node. With 15 thresholds, each feature is evaluated at
 * evenly spaced points across its global min/max range.
 */
#define NUM_THRESHOLDS 15

/*
 * MAX_DEPTH is the number of decision-node levels. MAX_DEPTH = 4 gives a root
 * level plus three more split levels before leaf predictions are made.
 */
#define MAX_DEPTH 4

/*
 * TREE_NODE_COUNT is derived from MAX_DEPTH for a complete binary tree stored
 * in an array using heap-style indexing: parent i has children 2*i+1 and 2*i+2.
 */
#define TREE_NODE_COUNT ((1 << MAX_DEPTH) - 1)

/*
 * MAX_LINE_LENGTH bounds one CSV line read from the Wine Quality file.
 */
#define MAX_LINE_LENGTH 2048

/*
 * DATASET_PATH is the default input file. A command-line path can override it.
 */
#define DATASET_PATH "dataset/winequality-white.csv"

/*
 * Human-readable names for the 11 Wine Quality input features. These names are
 * used only for output so split reports are understandable.
 */
static const char *FEATURE_NAMES[FEATURE_COUNT] = {
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
};

/*
 * Human-readable class labels matching quality_to_class().
 */
static const char *CLASS_NAMES[CLASS_COUNT] = {
    "low",
    "medium",
    "high"
};

/*
 * Struct: SplitNode
 * Purpose:
 *   Stores one decision-tree split node and its leaf predictions.
 * Fields:
 *   feature        - feature index used for the split
 *   threshold      - threshold used for feature <= threshold
 *   majority_class - fallback prediction for this node's full subset
 *   left_class     - majority prediction for rows going left
 *   right_class    - majority prediction for rows going right
 *   is_valid       - 1 if a useful split exists, 0 if this node is a leaf
 * MPI usage:
 *   None directly. Values are produced from MPI_Allreduce statistics.
 */
typedef struct {
    int feature;
    double threshold;
    int majority_class;
    int left_class;
    int right_class;
    int is_valid;
} SplitNode;

/*
 * Struct: DecisionTree
 * Purpose:
 *   Stores the complete tree as a flat array of SplitNode objects. The array
 *   representation avoids dynamic pointer allocation and keeps traversal simple.
 * MPI usage:
 *   None directly. Every rank builds the same tree after collective reductions.
 */
typedef struct {
    SplitNode nodes[TREE_NODE_COUNT];
} DecisionTree;

/*
 * Struct: PathCondition
 * Purpose:
 *   Describes one ancestor decision along a path from the root to a node.
 *   During recursive training, a row belongs to the current node only if it
 *   satisfies every PathCondition in the path.
 * MPI usage:
 *   None directly. It filters each rank's local rows before local statistics
 *   are combined with MPI_Allreduce.
 */
typedef struct {
    int feature;
    double threshold;
    int go_left;
} PathCondition;

/*
 * Function: trim_token
 * Purpose:
 *   Remove leading/trailing whitespace, double quotes, and line-ending
 *   characters from a CSV token.
 * Parameters:
 *   token - mutable token string returned by strtok
 * Returns:
 *   Pointer to the first useful character in the trimmed token.
 * MPI usage:
 *   None.
 */
static char *trim_token(char *token) {
    /* Wine Quality headers and fields may be quoted, so skip quotes here. */
    while (*token && (isspace((unsigned char)*token) || *token == '"')) {
        token++;
    }

    /* Trim trailing quotes, whitespace, CR, and LF in place. */
    char *end = token + strlen(token);
    while (end > token) {
        char c = *(end - 1);
        if (isspace((unsigned char)c) || c == '"' || c == '\r' || c == '\n') {
            end--;
            *end = '\0';
        } else {
            break;
        }
    }

    return token;
}

/*
 * Function: quality_to_class
 * Purpose:
 *   Convert the raw integer wine quality score into the three classes.
 * Parameters:
 *   quality - raw quality score from the CSV target column
 * Returns:
 *   0 for low, 1 for medium, 2 for high.
 * MPI usage:
 *   None.
 */
static int quality_to_class(int quality) {
    /* Low quality combines all scores up to and including 5. */
    if (quality <= 5) {
        return 0;
    }
    /* Medium quality is exactly score 6. */
    if (quality == 6) {
        return 1;
    }
    /* High quality combines all scores 7 and above. */
    return 2;
}

/*
 * Function: ensure_capacity
 * Purpose:
 *   Grow the rank-0 dataset arrays while reading an unknown number of CSV rows.
 * Parameters:
 *   features    - address of the flat feature array pointer
 *   labels      - address of the label array pointer
 *   capacity    - current allocated row capacity
 *   needed_rows - minimum number of rows that must fit
 * Returns:
 *   1 on success, 0 on allocation failure.
 * MPI usage:
 *   None.
 */
static int ensure_capacity(double **features, int **labels, int *capacity, int needed_rows) {
    if (needed_rows <= *capacity) {
        return 1;
    }

    /* Double capacity to keep CSV loading amortized linear instead of reallocating every row. */
    int new_capacity = (*capacity == 0) ? 1024 : (*capacity * 2);
    while (new_capacity < needed_rows) {
        new_capacity *= 2;
    }

    /* Features are stored flat: features[row * FEATURE_COUNT + feature]. */
    double *new_features = (double *)realloc(*features, (size_t)new_capacity * FEATURE_COUNT * sizeof(double));
    if (new_features == NULL) {
        return 0;
    }

    /* Labels are one integer class per row. */
    int *new_labels = (int *)realloc(*labels, (size_t)new_capacity * sizeof(int));
    if (new_labels == NULL) {
        *features = new_features;
        return 0;
    }

    *features = new_features;
    *labels = new_labels;
    *capacity = new_capacity;
    return 1;
}

/*
 * Function: read_wine_csv
 * Purpose:
 *   Read the semicolon-delimited Wine Quality CSV on rank 0, skip the header,
 *   store features in a flat double array, and convert labels into 3 classes.
 * Parameters:
 *   path         - CSV file path
 *   features_out - output pointer for flat feature data
 *   labels_out   - output pointer for integer class labels
 *   rows_out     - output row count
 * Returns:
 *   1 on success, 0 on parse or allocation failure.
 * MPI usage:
 *   None. This function is called by rank 0 before MPI_Bcast and MPI_Scatterv.
 */
static int read_wine_csv(const char *path, double **features_out, int **labels_out, int *rows_out) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return 0;
    }

    double *features = NULL;
    int *labels = NULL;
    int capacity = 0;
    int rows = 0;
    char line[MAX_LINE_LENGTH];

    /* The first line is a header with feature names, so it is read and ignored. */
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        return 0;
    }

    /* Each remaining line is one wine sample. */
    while (fgets(line, sizeof(line), file) != NULL) {
        if (strlen(line) <= 1) {
            continue;
        }

        if (!ensure_capacity(&features, &labels, &capacity, rows + 1)) {
            fclose(file);
            free(features);
            free(labels);
            return 0;
        }

        char *token = strtok(line, ";");
        int column = 0;

        /*
         * The Wine Quality CSV uses semicolons, not commas. The first 11 tokens
         * are feature values stored as:
         *   features[row * FEATURE_COUNT + column]
         * This flat layout is MPI-friendly because Scatterv can send contiguous
         * double blocks without packing structs.
         */
        while (token != NULL && column < FEATURE_COUNT) {
            features[(size_t)rows * FEATURE_COUNT + column] = strtod(trim_token(token), NULL);
            column++;
            token = strtok(NULL, ";");
        }

        /* After 11 feature columns, the next token must be the quality target. */
        if (column != FEATURE_COUNT || token == NULL) {
            fclose(file);
            free(features);
            free(labels);
            return 0;
        }

        /* Convert raw quality score to low/medium/high class immediately. */
        labels[rows] = quality_to_class(atoi(trim_token(token)));
        rows++;
    }

    fclose(file);
    *features_out = features;
    *labels_out = labels;
    *rows_out = rows;
    return 1;
}

/*
 * Function: row_matches_path
 * Purpose:
 *   Decide whether a local row belongs to the subtree currently being trained.
 * Parameters:
 *   row         - pointer to one feature row
 *   path        - ancestor split conditions from root to current node
 *   path_length - number of active conditions in path
 * Returns:
 *   1 if the row satisfies all path conditions, 0 otherwise.
 * MPI usage:
 *   None. This is local filtering before MPI_Allreduce combines statistics.
 */
static int row_matches_path(const double *row, const PathCondition *path, int path_length) {
    /*
     * Each PathCondition represents one previous branch decision. A row must
     * satisfy every ancestor condition to be counted in the current node.
     */
    for (int i = 0; i < path_length; i++) {
        double value = row[path[i].feature];

        /* Left branch means feature <= threshold; right branch means > threshold. */
        if (path[i].go_left) {
            if (value > path[i].threshold) {
                return 0;
            }
        } else if (value <= path[i].threshold) {
            return 0;
        }
    }

    return 1;
}

/*
 * Function: majority_class_from_counts
 * Purpose:
 *   Pick the class with the largest count. This is used for leaf predictions
 *   and for fallback predictions when no valid split is available.
 * Parameters:
 *   counts - class histogram for low, medium, and high
 * Returns:
 *   Majority class index.
 * MPI usage:
 *   None. The counts passed here may already be global from MPI_Allreduce.
 */
static int majority_class_from_counts(const long long counts[CLASS_COUNT]) {
    int best_class = 0;
    long long best_count = counts[0];

    for (int c = 1; c < CLASS_COUNT; c++) {
        if (counts[c] > best_count) {
            best_count = counts[c];
            best_class = c;
        }
    }

    return best_class;
}

/*
 * Function: gini_from_counts
 * Purpose:
 *   Compute Gini impurity for a class histogram.
 * Parameters:
 *   counts - class histogram
 *   total  - total samples represented by counts
 * Returns:
 *   Gini impurity, where lower is purer.
 * MPI usage:
 *   None. This operates on already-combined split counts.
 */
static double gini_from_counts(const long long counts[CLASS_COUNT], long long total) {
    if (total <= 0) {
        return 0.0;
    }

    /* Gini = 1 - sum(p_class^2). A pure node has Gini 0. */
    double impurity = 1.0;
    for (int c = 0; c < CLASS_COUNT; c++) {
        double p = (double)counts[c] / (double)total;
        impurity -= p * p;
    }
    return impurity;
}

/*
 * Function: make_empty_node
 * Purpose:
 *   Create a default invalid node that behaves like a leaf until trained.
 * Parameters:
 *   None.
 * Returns:
 *   SplitNode initialized with safe defaults.
 * MPI usage:
 *   None.
 */
static SplitNode make_empty_node(void) {
    SplitNode node;
    node.feature = -1;
    node.threshold = 0.0;
    node.majority_class = 0;
    node.left_class = 0;
    node.right_class = 0;
    node.is_valid = 0;
    return node;
}

/*
 * Function: compute_global_class_counts
 * Purpose:
 *   Count low/medium/high labels on each rank and combine them into a global
 *   class distribution available to every rank.
 * Parameters:
 *   local_labels  - labels owned by this MPI rank
 *   local_rows    - number of local labels
 *   global_counts - output global class counts
 *   comm          - MPI communicator
 * Returns:
 *   None.
 * MPI usage:
 *   MPI_Allreduce sums local class histograms. All ranks receive the global
 *   result, which keeps output and training context consistent.
 */
static void compute_global_class_counts(const int *local_labels,
                                        int local_rows,
                                        long long global_counts[CLASS_COUNT],
                                        MPI_Comm comm) {
    long long local_counts[CLASS_COUNT] = {0, 0, 0};

    /* Each rank counts only its scattered partition. */
    for (int i = 0; i < local_rows; i++) {
        local_counts[local_labels[i]]++;
    }

    /*
     * MPI_Allreduce gives every rank the full class distribution. Rank 0 prints
     * it as dataset context while keeping the count calculation distributed.
     */
    MPI_Allreduce(local_counts, global_counts, CLASS_COUNT, MPI_LONG_LONG, MPI_SUM, comm);
}

/*
 * Function: train_best_split
 * Purpose:
 *   Train one decision-tree node by evaluating all feature/threshold candidates
 *   on distributed data and selecting the split with the lowest weighted Gini.
 * Parameters:
 *   local_features - flat feature matrix partition owned by this rank
 *   local_labels   - label partition owned by this rank
 *   local_rows     - number of rows in this rank's partition
 *   path           - ancestor branch conditions defining this node's subset
 *   path_length    - number of ancestor conditions
 *   comm           - MPI communicator
 * Returns:
 *   A SplitNode containing the best split and majority-class leaf predictions.
 * MPI usage:
 *   MPI_Allreduce combines local class counts, feature min/max values, and
 *   split statistics. Every rank receives the same global statistics and
 *   therefore builds the same split without rank 0 training serially.
 */
static SplitNode train_best_split(const double *local_features,
                                  const int *local_labels,
                                  int local_rows,
                                  const PathCondition *path,
                                  int path_length,
                                  MPI_Comm comm) {
    SplitNode node;
    node.feature = -1;
    node.threshold = 0.0;
    node.majority_class = 0;
    node.left_class = 0;
    node.right_class = 0;
    node.is_valid = 0;

    long long local_node_counts[CLASS_COUNT] = {0, 0, 0};
    double local_min[FEATURE_COUNT];
    double local_max[FEATURE_COUNT];

    /* Start local feature ranges at extremes so real local values replace them. */
    for (int f = 0; f < FEATURE_COUNT; f++) {
        local_min[f] = DBL_MAX;
        local_max[f] = -DBL_MAX;
    }

    /*
     * First local pass:
     *   - filter rows that belong to this tree node
     *   - count local class labels
     *   - compute local min/max for every feature
     */
    for (int i = 0; i < local_rows; i++) {
        /*
         * Flat indexing: row i begins at local_features[i * FEATURE_COUNT].
         * This is the local partition, not the full dataset.
         */
        const double *row = &local_features[(size_t)i * FEATURE_COUNT];
        if (!row_matches_path(row, path, path_length)) {
            continue;
        }

        int label = local_labels[i];
        local_node_counts[label]++;

        for (int f = 0; f < FEATURE_COUNT; f++) {
            if (row[f] < local_min[f]) {
                local_min[f] = row[f];
            }
            if (row[f] > local_max[f]) {
                local_max[f] = row[f];
            }
        }
    }

    long long global_node_counts[CLASS_COUNT];
    double global_min[FEATURE_COUNT];
    double global_max[FEATURE_COUNT];

    /*
     * MPI_Allreduce keeps every rank synchronized with the node's global class
     * histogram, so majority-class leaves are based on all distributed rows.
     */
    MPI_Allreduce(local_node_counts, global_node_counts, CLASS_COUNT, MPI_LONG_LONG, MPI_SUM, comm);

    /*
     * MPI_Allreduce also combines feature min/max values across ranks. These
     * global ranges define the evenly spaced threshold candidates.
     */
    MPI_Allreduce(local_min, global_min, FEATURE_COUNT, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(local_max, global_max, FEATURE_COUNT, MPI_DOUBLE, MPI_MAX, comm);

    /* Total rows in this node after all ranks have contributed counts. */
    long long total_rows = 0;
    for (int c = 0; c < CLASS_COUNT; c++) {
        total_rows += global_node_counts[c];
    }

    if (total_rows == 0) {
        return node;
    }

    /* Majority class is the fallback prediction if the node cannot split. */
    node.majority_class = majority_class_from_counts(global_node_counts);
    node.left_class = node.majority_class;
    node.right_class = node.majority_class;

    /* If the node already contains one class only, it is pure and should stay a leaf. */
    int non_empty_classes = 0;
    for (int c = 0; c < CLASS_COUNT; c++) {
        if (global_node_counts[c] > 0) {
            non_empty_classes++;
        }
    }

    if (non_empty_classes <= 1) {
        node.is_valid = 0;
        return node;
    }

    const int threshold_count = FEATURE_COUNT * NUM_THRESHOLDS;
    const int stats_count = threshold_count * 2 * CLASS_COUNT;
    /*
     * local_stats/global_stats layout:
     *   feature -> threshold candidate -> split side(left/right) -> class
     * This stores histograms for every possible split candidate.
     */
    long long *local_stats = (long long *)calloc((size_t)stats_count, sizeof(long long));
    long long *global_stats = (long long *)calloc((size_t)stats_count, sizeof(long long));

    if (local_stats == NULL || global_stats == NULL) {
        free(local_stats);
        free(global_stats);
        return node;
    }

    /*
     * Second local pass:
     *   For each local row in this node, test every feature at every candidate
     *   threshold and increment the appropriate left/right class histogram.
     */
    for (int i = 0; i < local_rows; i++) {
        const double *row = &local_features[(size_t)i * FEATURE_COUNT];
        if (!row_matches_path(row, path, path_length)) {
            continue;
        }

        int label = local_labels[i];
        for (int f = 0; f < FEATURE_COUNT; f++) {
            double range = global_max[f] - global_min[f];
            for (int q = 0; q < NUM_THRESHOLDS; q++) {
                /*
                 * Candidate thresholds are evenly spaced inside the global
                 * feature range. With NUM_THRESHOLDS = 15, this tests 1/16,
                 * 2/16, ..., 15/16 of the min/max interval.
                 */
                double fraction = (double)(q + 1) / (NUM_THRESHOLDS + 1.0);
                double threshold = global_min[f] + fraction * range;
                int side = (row[f] <= threshold) ? 0 : 1;
                int index = ((f * NUM_THRESHOLDS + q) * 2 + side) * CLASS_COUNT + label;
                local_stats[index]++;
            }
        }
    }

    /*
     * MPI_Allreduce merges split statistics for every feature/threshold pair.
     * Each rank can then independently choose the same lowest-Gini split.
     */
    MPI_Allreduce(local_stats, global_stats, stats_count, MPI_LONG_LONG, MPI_SUM, comm);

    double best_gini = DBL_MAX;
    int best_feature = -1;
    int best_candidate = -1;
    double best_threshold = 0.0;

    /*
     * Evaluate all globally combined candidate histograms. Since every rank has
     * the same global_stats after Allreduce, every rank chooses the same split.
     */
    for (int f = 0; f < FEATURE_COUNT; f++) {
        double range = global_max[f] - global_min[f];
        if (range <= 0.0) {
            continue;
        }

        for (int q = 0; q < NUM_THRESHOLDS; q++) {
            long long left_counts[CLASS_COUNT];
            long long right_counts[CLASS_COUNT];
            long long left_total = 0;
            long long right_total = 0;

            for (int c = 0; c < CLASS_COUNT; c++) {
                int left_index = ((f * NUM_THRESHOLDS + q) * 2) * CLASS_COUNT + c;
                int right_index = ((f * NUM_THRESHOLDS + q) * 2 + 1) * CLASS_COUNT + c;
                left_counts[c] = global_stats[left_index];
                right_counts[c] = global_stats[right_index];
                left_total += left_counts[c];
                right_total += right_counts[c];
            }

            if (left_total == 0 || right_total == 0) {
                continue;
            }

            /* Compute impurity for the candidate's left and right partitions. */
            double left_gini = gini_from_counts(left_counts, left_total);
            double right_gini = gini_from_counts(right_counts, right_total);
            /*
             * Weighted Gini gives larger child nodes proportionally more
             * influence. The best split is the candidate with the smallest
             * weighted impurity after combining all MPI ranks.
             */
            double weighted_gini = ((double)left_total / (double)total_rows) * left_gini
                                 + ((double)right_total / (double)total_rows) * right_gini;

            /* Keep the lowest-Gini feature/threshold candidate seen so far. */
            if (weighted_gini < best_gini) {
                best_gini = weighted_gini;
                best_feature = f;
                best_candidate = q;
                best_threshold = global_min[f] + ((double)(q + 1) / (NUM_THRESHOLDS + 1.0)) * range;
            }
        }
    }

    if (best_feature >= 0) {
        long long left_counts[CLASS_COUNT];
        long long right_counts[CLASS_COUNT];

        /* Retrieve the final left/right histograms for the selected split. */
        for (int c = 0; c < CLASS_COUNT; c++) {
            int left_index = ((best_feature * NUM_THRESHOLDS + best_candidate) * 2) * CLASS_COUNT + c;
            int right_index = ((best_feature * NUM_THRESHOLDS + best_candidate) * 2 + 1) * CLASS_COUNT + c;
            left_counts[c] = global_stats[left_index];
            right_counts[c] = global_stats[right_index];
        }

        node.feature = best_feature;
        node.threshold = best_threshold;
        /*
         * Leaf prediction is majority class. At maximum depth, prediction uses
         * left_class or right_class depending on the final branch direction.
         */
        node.left_class = majority_class_from_counts(left_counts);
        node.right_class = majority_class_from_counts(right_counts);
        node.is_valid = 1;
    }

    free(local_stats);
    free(global_stats);
    return node;
}

/*
 * Function: initialize_tree
 * Purpose:
 *   Fill every array slot with an invalid leaf-like node before training.
 * Parameters:
 *   tree - decision tree to initialize
 * Returns:
 *   None.
 * MPI usage:
 *   None.
 */
static void initialize_tree(DecisionTree *tree) {
    for (int i = 0; i < TREE_NODE_COUNT; i++) {
        tree->nodes[i] = make_empty_node();
    }
}

/*
 * Function: train_tree_node
 * Purpose:
 *   Recursively train one node and then train its left and right children until
 *   MAX_DEPTH is reached or the node becomes a leaf.
 * Parameters:
 *   tree           - tree being trained
 *   node_index     - array index of the current node
 *   depth          - current tree depth
 *   path           - ancestor decisions needed to filter rows for this node
 *   path_length    - number of active ancestor decisions
 *   local_features - feature partition owned by this rank
 *   local_labels   - label partition owned by this rank
 *   local_rows     - number of rows owned by this rank
 *   comm           - MPI communicator
 * Returns:
 *   None.
 * MPI usage:
 *   Calls train_best_split(), which uses MPI_Allreduce. All ranks recursively
 *   visit the same node indexes and therefore build matching trees.
 */
static void train_tree_node(DecisionTree *tree,
                            int node_index,
                            int depth,
                            PathCondition *path,
                            int path_length,
                            const double *local_features,
                            const int *local_labels,
                            int local_rows,
                            MPI_Comm comm) {
    if (node_index >= TREE_NODE_COUNT || depth >= MAX_DEPTH) {
        return;
    }

    tree->nodes[node_index] = train_best_split(local_features,
                                               local_labels,
                                               local_rows,
                                               path,
                                               path_length,
                                               comm);

    if (!tree->nodes[node_index].is_valid || depth == MAX_DEPTH - 1) {
        return;
    }

    /*
     * Extend the current path with this node's split. Child training filters
     * local rows using this path, so each child receives only the rows that
     * would reach that child in a serial tree.
     */
    PathCondition next_path[MAX_DEPTH];

    for (int i = 0; i < path_length; i++) {
        next_path[i] = path[i];
    }

    next_path[path_length].feature = tree->nodes[node_index].feature;
    next_path[path_length].threshold = tree->nodes[node_index].threshold;
    /* Train left child: rows with feature <= threshold. */
    next_path[path_length].go_left = 1;
    train_tree_node(tree,
                    2 * node_index + 1,
                    depth + 1,
                    next_path,
                    path_length + 1,
                    local_features,
                    local_labels,
                    local_rows,
                    comm);

    /* Train right child: rows with feature > threshold. */
    next_path[path_length].go_left = 0;
    train_tree_node(tree,
                    2 * node_index + 2,
                    depth + 1,
                    next_path,
                    path_length + 1,
                    local_features,
                    local_labels,
                    local_rows,
                    comm);
}

/*
 * Function: train_tree
 * Purpose:
 *   Train the full fixed-depth decision tree from the root.
 * Parameters:
 *   local_features - feature partition owned by this rank
 *   local_labels   - label partition owned by this rank
 *   local_rows     - number of rows owned by this rank
 *   comm           - MPI communicator
 * Returns:
 *   Fully trained DecisionTree. Every rank receives the same tree structure
 *   because every split is chosen from MPI_Allreduce global statistics.
 * MPI usage:
 *   Indirectly uses MPI_Allreduce through train_tree_node()/train_best_split().
 */
static DecisionTree train_tree(const double *local_features,
                               const int *local_labels,
                               int local_rows,
                               MPI_Comm comm) {
    DecisionTree tree;
    PathCondition path[MAX_DEPTH];

    initialize_tree(&tree);
    train_tree_node(&tree, 0, 0, path, 0, local_features, local_labels, local_rows, comm);

    return tree;
}

/*
 * Function: predict_row
 * Purpose:
 *   Traverse the trained tree for one row and return a predicted class.
 * Parameters:
 *   tree - trained decision tree
 *   row  - one feature row
 * Returns:
 *   Predicted class index.
 * MPI usage:
 *   None. Prediction is local on each rank's scattered rows.
 */
static int predict_row(const DecisionTree *tree, const double *row) {
    int node_index = 0;

    /* Walk down the array-based tree until a leaf or the maximum depth. */
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        const SplitNode *node = &tree->nodes[node_index];

        /* Invalid nodes are leaves; predict their stored majority class. */
        if (!node->is_valid) {
            return node->majority_class;
        }

        int go_left = row[node->feature] <= node->threshold;

        /*
         * At the last split level, the children are leaf predictions stored in
         * the current node rather than separate tree nodes.
         */
        if (depth == MAX_DEPTH - 1) {
            return go_left ? node->left_class : node->right_class;
        }

        /* Heap-style child indexing: left = 2*i+1, right = 2*i+2. */
        node_index = go_left ? (2 * node_index + 1) : (2 * node_index + 2);
        if (node_index >= TREE_NODE_COUNT) {
            return go_left ? node->left_class : node->right_class;
        }
    }

    return tree->nodes[0].majority_class;
}

/*
 * Function: local_correct_predictions
 * Purpose:
 *   Predict labels for this rank's local rows and count correct predictions.
 * Parameters:
 *   tree           - trained decision tree
 *   local_features - feature partition owned by this rank
 *   local_labels   - label partition owned by this rank
 *   local_rows     - number of rows owned by this rank
 * Returns:
 *   Number of correct predictions on this rank.
 * MPI usage:
 *   None directly. The returned count is later summed with MPI_Reduce.
 */
static int local_correct_predictions(const DecisionTree *tree,
                                     const double *local_features,
                                     const int *local_labels,
                                     int local_rows) {
    int correct = 0;

    for (int i = 0; i < local_rows; i++) {
        /* Each rank predicts only the rows it received from MPI_Scatterv. */
        const double *row = &local_features[(size_t)i * FEATURE_COUNT];
        int prediction = predict_row(tree, row);
        if (prediction == local_labels[i]) {
            correct++;
        }
    }

    return correct;
}

/*
 * Function: print_tree_node
 * Purpose:
 *   Print one trained split node using feature names, or print leaf fallback
 *   information when no valid split exists.
 * Parameters:
 *   label - human-readable node location string
 *   node  - node to print
 * Returns:
 *   None.
 * MPI usage:
 *   None. Called by rank 0 after training.
 */
static void print_tree_node(const char *label, const SplitNode *node) {
    if (node->is_valid && node->feature >= 0 && node->feature < FEATURE_COUNT) {
        printf("%s: feature %d (%s), threshold %.6f\n",
               label,
               node->feature,
               FEATURE_NAMES[node->feature],
               node->threshold);
    } else {
        printf("%s: leaf only, majority class %d (%s)\n",
               label,
               node->majority_class,
               CLASS_NAMES[node->majority_class]);
    }
}

/*
 * Function: build_node_label
 * Purpose:
 *   Convert an array index into a readable tree path such as
 *   "Depth 3 left-right-left node".
 * Parameters:
 *   node_index - index in the array-based tree
 *   label      - output string buffer
 *   label_size - size of output buffer
 * Returns:
 *   None.
 * MPI usage:
 *   None.
 */
static void build_node_label(int node_index, char *label, size_t label_size) {
    if (node_index == 0) {
        snprintf(label, label_size, "Depth 0 root node");
        return;
    }

    int directions[MAX_DEPTH];
    int depth = 0;
    int current = node_index;

    /* Walk from child to root to recover left/right directions. */
    while (current > 0 && depth < MAX_DEPTH) {
        int parent = (current - 1) / 2;
        directions[depth] = (current == (2 * parent + 1)) ? 0 : 1;
        current = parent;
        depth++;
    }

    char path[128] = "";
    /* Reverse the collected directions so the label reads from root to node. */
    for (int i = depth - 1; i >= 0; i--) {
        if (path[0] != '\0') {
            strncat(path, "-", sizeof(path) - strlen(path) - 1);
        }
        strncat(path, directions[i] == 0 ? "left" : "right", sizeof(path) - strlen(path) - 1);
    }

    snprintf(label, label_size, "Depth %d %s node", depth, path);
}

/*
 * Function: print_tree
 * Purpose:
 *   Print every decision node in the array-based tree.
 * Parameters:
 *   tree - trained tree to print
 * Returns:
 *   None.
 * MPI usage:
 *   None. Called only by rank 0 for final output.
 */
static void print_tree(const DecisionTree *tree) {
    for (int i = 0; i < TREE_NODE_COUNT; i++) {
        char label[160];
        build_node_label(i, label, sizeof(label));
        print_tree_node(label, &tree->nodes[i]);
    }
}

/*
 * Function: build_counts_and_displacements
 * Purpose:
 *   Build Scatterv counts and displacements for uneven row distribution.
 * Parameters:
 *   total_rows     - total number of dataset rows
 *   processes      - MPI world size
 *   row_counts     - output row count per rank
 *   row_displs     - output row displacement per rank
 *   feature_counts - output feature-value count per rank
 *   feature_displs - output feature-value displacement per rank
 * Returns:
 *   None.
 * MPI usage:
 *   None directly. The arrays produced here are used by MPI_Scatterv.
 */
static void build_counts_and_displacements(int total_rows,
                                           int processes,
                                           int *row_counts,
                                           int *row_displs,
                                           int *feature_counts,
                                           int *feature_displs) {
    int base = total_rows / processes;
    int remainder = total_rows % processes;
    int row_offset = 0;
    int feature_offset = 0;

    /*
     * Rows may not divide evenly by process count. The first "remainder" ranks
     * receive one extra row, which is why MPI_Scatterv is required instead of
     * MPI_Scatter. Scatterv supports different send counts per process.
     */
    for (int rank = 0; rank < processes; rank++) {
        row_counts[rank] = base + (rank < remainder ? 1 : 0);
        row_displs[rank] = row_offset;
        /*
         * Feature data is flat, so each row contributes FEATURE_COUNT doubles.
         * Counts/displacements for feature Scatterv are measured in doubles.
         */
        feature_counts[rank] = row_counts[rank] * FEATURE_COUNT;
        feature_displs[rank] = feature_offset;

        row_offset += row_counts[rank];
        feature_offset += feature_counts[rank];
    }
}

/*
 * Function: main
 * Purpose:
 *   Coordinate MPI initialization, dataset loading, data distribution, tree
 *   training, prediction, timing, and final reporting.
 * Parameters:
 *   argc - command-line argument count
 *   argv - command-line arguments; argv[1] can override DATASET_PATH
 * Returns:
 *   0 on success, 1 on dataset or allocation failure.
 * MPI usage:
 *   MPI_Init/MPI_Finalize manage MPI lifetime.
 *   MPI_Bcast shares metadata from rank 0.
 *   MPI_Scatterv distributes feature and label partitions.
 *   MPI_Gather reports local row counts.
 *   MPI_Allreduce is used during class counting and training.
 *   MPI_Barrier and MPI_Wtime measure parallel wall-clock time.
 *   MPI_Reduce combines elapsed time and correct-prediction counts on rank 0.
 */
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    /* All ranks learn their identity and the total number of MPI processes. */
    int rank = 0;
    int processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int total_rows = 0;
    const int feature_count = FEATURE_COUNT;
    const int class_count = CLASS_COUNT;
    const char *dataset_path = (argc > 1) ? argv[1] : DATASET_PATH;
    double *all_features = NULL;
    int *all_labels = NULL;

    /*
     * Rank 0 is the only process that reads the full CSV file. At this point,
     * worker ranks know nothing about the dataset contents; they will receive
     * metadata through MPI_Bcast and data partitions through MPI_Scatterv.
     */
    if (rank == 0) {
        if (!read_wine_csv(dataset_path, &all_features, &all_labels, &total_rows)) {
            fprintf(stderr, "Failed to read %s\n", dataset_path);
            total_rows = -1;
        }
    }

    /*
     * MPI_Bcast sends metadata read by rank 0 to every process. All ranks need
     * the row count to allocate receive buffers and participate in Scatterv.
     * Worker ranks receive total_rows here before any dataset rows are sent.
     */
    MPI_Bcast(&total_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (total_rows <= 0) {
        free(all_features);
        free(all_labels);
        MPI_Finalize();
        return 1;
    }

    int metadata[2] = {feature_count, class_count};
    /* Broadcast feature and class counts so every rank has the same metadata. */
    MPI_Bcast(metadata, 2, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * Rank 0 prepares counts/displacements for Scatterv, but every rank builds
     * the same arrays because all ranks know total_rows and processes. This
     * makes local_rows calculation simple and consistent on every rank.
     */
    int *row_counts = (int *)malloc((size_t)processes * sizeof(int));
    int *row_displs = (int *)malloc((size_t)processes * sizeof(int));
    int *feature_counts = (int *)malloc((size_t)processes * sizeof(int));
    int *feature_displs = (int *)malloc((size_t)processes * sizeof(int));

    if (row_counts == NULL || row_displs == NULL || feature_counts == NULL || feature_displs == NULL) {
        fprintf(stderr, "Rank %d failed to allocate Scatterv metadata\n", rank);
        free(all_features);
        free(all_labels);
        free(row_counts);
        free(row_displs);
        free(feature_counts);
        free(feature_displs);
        MPI_Finalize();
        return 1;
    }

    build_counts_and_displacements(total_rows, processes, row_counts, row_displs, feature_counts, feature_displs);

    /*
     * local_rows is the number of dataset rows assigned to this rank. The value
     * can differ by one row across ranks when total_rows is not divisible by
     * processes.
     */
    int local_rows = row_counts[rank];
    int *gathered_row_counts = NULL;

    /*
     * Only rank 0 needs the receive buffer for MPI_Gather because only rank 0
     * prints the row distribution summary.
     */
    if (rank == 0) {
        gathered_row_counts = (int *)malloc((size_t)processes * sizeof(int));
        if (gathered_row_counts == NULL) {
            fprintf(stderr, "Rank 0 failed to allocate gathered row counts\n");
            free(all_features);
            free(all_labels);
            free(row_counts);
            free(row_displs);
            free(feature_counts);
            free(feature_displs);
            MPI_Finalize();
            return 1;
        }
    }

    /*
     * MPI_Gather collects each rank's local_rows value on rank 0. This does not
     * affect training; it is used only to show the Scatterv row distribution.
     */
    MPI_Gather(&local_rows, 1, MPI_INT,
               gathered_row_counts, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    /*
     * Each rank allocates receive buffers only for its own partition. Worker
     * ranks never store the full dataset.
     */
    double *local_features = (double *)malloc((size_t)local_rows * FEATURE_COUNT * sizeof(double));
    int *local_labels = (int *)malloc((size_t)local_rows * sizeof(int));

    if ((local_rows > 0 && local_features == NULL) || (local_rows > 0 && local_labels == NULL)) {
        fprintf(stderr, "Rank %d failed to allocate local data\n", rank);
        free(all_features);
        free(all_labels);
        free(row_counts);
        free(row_displs);
        free(feature_counts);
        free(feature_displs);
        free(gathered_row_counts);
        free(local_features);
        free(local_labels);
        MPI_Finalize();
        return 1;
    }

    /*
     * MPI_Scatterv distributes uneven partitions correctly. Feature rows are
     * flattened as doubles, so counts/displacements are measured in doubles.
     * Rank 0 sends from all_features; every rank receives only local_features.
     */
    MPI_Scatterv(all_features, feature_counts, feature_displs, MPI_DOUBLE,
                 local_features, local_rows * FEATURE_COUNT, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /*
     * Labels are scattered with row counts/displacements because there is one
     * integer label per dataset row.
     */
    MPI_Scatterv(all_labels, row_counts, row_displs, MPI_INT,
                 local_labels, local_rows, MPI_INT,
                 0, MPI_COMM_WORLD);

    /*
     * All ranks participate in computing global class counts. Each rank counts
     * its own labels, then MPI_Allreduce combines the counts and returns the
     * global distribution to every rank.
     */
    long long global_class_counts[CLASS_COUNT];
    compute_global_class_counts(local_labels, local_rows, global_class_counts, MPI_COMM_WORLD);

    /*
     * Synchronize before timing so slow setup on one process does not make
     * another process start the measured section early. MPI_Wtime reports
     * wall-clock time, which is the usual timing measure for parallel programs.
     */
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /*
     * Training is distributed: every rank evaluates local rows and uses
     * MPI_Allreduce inside train_tree to combine statistics at each node.
     */
    DecisionTree tree = train_tree(local_features, local_labels, local_rows, MPI_COMM_WORLD);

    /*
     * Prediction is local. Each rank predicts only its partition and counts how
     * many of those local predictions are correct.
     */
    int local_correct = local_correct_predictions(&tree, local_features, local_labels, local_rows);

    double local_elapsed = MPI_Wtime() - start_time;
    double elapsed = 0.0;
    /*
     * Parallel elapsed time is the slowest rank's measured time. MPI_Reduce
     * with MPI_MAX sends that maximum time to rank 0 for reporting.
     */
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int global_correct = 0;

    /*
     * MPI_Reduce gathers the local correct-prediction counts on rank 0, where
     * the final accuracy is printed.
     * Accuracy = global correct predictions / total dataset rows.
     */
    MPI_Reduce(&local_correct, &global_correct, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    /*
     * Rank 0 owns the final reporting responsibility. Worker ranks participate
     * in computation and collectives, but they do not print the final summary.
     */
    if (rank == 0) {
        double accuracy = 100.0 * (double)global_correct / (double)total_rows;

        printf("Dataset: Wine Quality\n");
        printf("Rows: %d\n", total_rows);
        printf("Features: %d\n", FEATURE_COUNT);
        printf("Classes: %d\n", CLASS_COUNT);
        printf("Processes: %d\n", processes);
        printf("Row distribution:\n");
        for (int p = 0; p < processes; p++) {
            printf("  Process %d: %d rows\n", p, gathered_row_counts[p]);
        }
        printf("Class distribution:\n");
        for (int c = 0; c < CLASS_COUNT; c++) {
            printf("  %s: %lld\n", CLASS_NAMES[c], global_class_counts[c]);
        }
        print_tree(&tree);
        printf("Accuracy: %.2f %%\n", accuracy);
        printf("Execution time: %.6f seconds\n", elapsed);
    }

    free(all_features);
    free(all_labels);
    free(row_counts);
    free(row_displs);
    free(feature_counts);
    free(feature_displs);
    free(gathered_row_counts);
    free(local_features);
    free(local_labels);

    MPI_Finalize();
    return 0;
}

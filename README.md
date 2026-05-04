# MPI Decision Tree - Wine Quality

![Language](https://img.shields.io/badge/Language-C-blue)
![MPI](https://img.shields.io/badge/MPI-MS--MPI-green)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-Wine%20Quality-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Parallel Computing](https://img.shields.io/badge/Parallel%20Computing-MPI-orange)
![Model](https://img.shields.io/badge/Model-Decision%20Tree-red)

This project implements an MPI Decision Tree classifier in C using MS-MPI on Windows and the Wine Quality dataset (`winequality-white.csv`).

## Dataset

- File: `dataset/winequality-white.csv`
- Delimiter: semicolon (`;`)
- Header row: yes
- Rows: 4,898
- Input features: 11 numeric wine measurements
- Target: `quality`
- Converted classes:
  - Class 0, low: `quality <= 5`
  - Class 1, medium: `quality == 6`
  - Class 2, high: `quality >= 7`

The dataset is split deterministically into:

- Training set: first 80% of rows, 3,918 rows
- Test set: last 20% of rows, 980 rows

The model trains only on the training set. Accuracy is reported only on the held-out test set.

## From Iris To Wine Quality

The Iris dataset usually has 150 rows, 4 features, clean class separation, and 3 balanced species classes. The Wine Quality dataset has 4,898 rows, 11 features, and more class overlap.

Wine Quality is larger and noisier than Iris because the target labels come from subjective quality scores, and wines with neighboring quality values can have very similar chemical measurements. Moderate test accuracy is expected because the classes are less separable.

This implementation handles Wine Quality by:

- Reading a semicolon-separated CSV file.
- Skipping the CSV header row.
- Storing 11 features in flat `double` arrays for MPI communication.
- Converting numeric quality scores into low, medium, and high classes.
- Using uneven row partitioning with `MPI_Scatterv`, because train and test row counts may not divide evenly by the process count.

## Model Configuration

The current MPI Decision Tree uses:

- `MAX_DEPTH = 4`
- `NUM_THRESHOLDS = 15`
- `TREE_NODE_COUNT = ((1 << MAX_DEPTH) - 1)`
- Gini impurity for split selection

The implementation uses `MAX_DEPTH = 4`, which creates decision nodes from depth 0 through depth 3. With this array-based complete binary tree layout, there are 15 decision nodes and 16 majority-class leaf predictions.

Each feature evaluates 15 candidate thresholds distributed across the feature range:

```text
min + ((q + 1) / (NUM_THRESHOLDS + 1.0)) * (max - min)
```

for `q = 0` to `NUM_THRESHOLDS - 1`.

Increasing `MAX_DEPTH` to 4 and `NUM_THRESHOLDS` to 15 improves model capacity. The tradeoff is higher computation time because more tree nodes and more candidate thresholds are evaluated. Wine Quality remains noisy and less separable than Iris, so test accuracy is still expected to be moderate.

## MPI Design

Rank 0 reads the full CSV file, creates the deterministic 80/20 split, and initially owns both full arrays:

- Features in a flat `double` array.
- Labels in a flat `int` array.

The MPI collective functions used are:

- `MPI_Bcast`: broadcasts dataset metadata such as total row count, train row count, test row count, feature count, and class count from rank 0 to all processes.
- `MPI_Scatterv`: distributes uneven training and test row partitions to all processes. This is used instead of `MPI_Scatter` because row counts may differ by process.
- `MPI_Gather`: collects each process's local train/test row counts on rank 0 for the row distribution output.
- `MPI_Allreduce`: combines training class counts, feature min/max values, and split statistics during Gini impurity calculation. It also combines train/test class distributions for reporting.
- `MPI_Reduce`: combines local test correct prediction counts into the final global test accuracy result, and also reduces elapsed time using the maximum rank time.

`MPI_Barrier` synchronizes ranks before timing starts, and `MPI_Wtime` measures wall-clock execution time for one complete train + test prediction cycle. CSV reading is outside the timed section.

## Decision Tree

The classifier uses Gini impurity:

1. Rank-local training rows are filtered to the current tree node.
2. Each rank computes local training class counts, feature min/max values, and split histograms.
3. `MPI_Allreduce` combines those local statistics into global training statistics.
4. Each rank independently selects the same lowest weighted-Gini split.
5. Prediction traverses the trained tree on local test rows only.
6. `MPI_Reduce` combines local test correct counts into global test accuracy.

Tree output prints decision nodes from depth 0 through depth 3.

## Compile On Windows With MS-MPI

Open a Developer Command Prompt or a terminal where MS-MPI and the C compiler are available.

Using Microsoft C compiler:

```bat
cl /I"%MSMPI_INC%" src\main.c /link /LIBPATH:"%MSMPI_LIB64%" msmpi.lib /OUT:main.exe
```

If your MS-MPI installation uses explicit paths, an example is:

```bat
cl /I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" src\main.c /link /LIBPATH:"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" msmpi.lib /OUT:main.exe
```

## Run

Run from the project root so the default dataset path can be found. The dataset path argument is optional and defaults to:

```text
dataset/winequality-white.csv
```

Run with the default dataset path:

```bat
mpiexec -n 1 main.exe
mpiexec -n 2 main.exe
mpiexec -n 4 main.exe
mpiexec -n 8 main.exe
```

Run with an explicit dataset path:

```bat
mpiexec -n 4 main.exe dataset\winequality-white.csv
```

## Example Terminal Output

```text
Dataset: Wine Quality
Total rows: 4898
Train rows: 3918
Test rows: 980
Features: 11
Classes: 3
Processes: 4

Train row distribution:
  Process 0: 980 rows
  Process 1: 980 rows
  Process 2: 979 rows
  Process 3: 979 rows

Test row distribution:
  Process 0: 245 rows
  Process 1: 245 rows
  Process 2: 245 rows
  Process 3: 245 rows

Training class distribution:
  low: 1348
  medium: 1681
  high: 889

Test class distribution:
  low: 292
  medium: 517
  high: 171

Tree configuration:
  MAX_DEPTH: 4
  NUM_THRESHOLDS: 15

Depth 0 root node: feature 10 (alcohol), threshold 10.625000
Depth 1 left node: feature 1 (volatile acidity), threshold 0.253437
Depth 1 right node: feature 10 (alcohol), threshold 11.731250
Depth 2 left-left node: feature 9 (sulphates), threshold 0.495625
Depth 2 left-right node: feature 10 (alcohol), threshold 9.775000
Depth 2 right-left node: feature 5 (free sulfur dioxide), threshold 11.031250
Depth 2 right-right node: feature 0 (fixed acidity), threshold 7.575000
Depth 3 left-left-left node: feature 10 (alcohol), threshold 8.812500
Depth 3 left-left-right node: feature 9 (sulphates), threshold 0.881875
Depth 3 left-right-left node: feature 1 (volatile acidity), threshold 0.395000
Depth 3 left-right-right node: feature 5 (free sulfur dioxide), threshold 19.875000
Depth 3 right-left-left node: feature 4 (chlorides), threshold 0.045500
Depth 3 right-left-right node: feature 1 (volatile acidity), threshold 0.475000
Depth 3 right-right-left node: feature 5 (free sulfur dioxide), threshold 8.812500
Depth 3 right-right-right node: feature 1 (volatile acidity), threshold 0.387500
Test Accuracy: 55.41 %
Execution time: 0.004737 seconds
```

Exact timing can vary by machine, compiler, background load, and process placement.

## Performance, Speedup, And Scalability

Speedup measures how much faster the parallel version is compared with one process:

```text
speedup = T1 / Tp
```

Efficiency measures how well the program uses `p` processes:

```text
efficiency = speedup / p
```

Execution time for one complete train + test prediction cycle:

| Processes | Time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 0.013953 | 1.00 | 1.00 |
| 2 | 0.008140 | 1.71 | 0.86 |
| 4 | 0.006306 | 2.21 | 0.55 |
| 8 | 0.004737 | 2.95 | 0.37 |

![Speedup Analysis](images/speedup_graph.svg)

Figure 1: Speedup analysis of the MPI Decision Tree implementation.

![Efficiency Analysis](images/efficiency_graph.svg)

Figure 2: Efficiency analysis of the MPI Decision Tree implementation.

Scalability means how well the MPI Decision Tree continues to reduce runtime as more processes are added.

## Performance Interpretation

Execution time decreases as the number of MPI processes increases in the measured local results. The 8-process run gives the best runtime at 0.004737 seconds, reaching 2.95x speedup compared with 1 process.

Efficiency decreases as process count increases. This means the implementation benefits from parallel execution, but scaling is sublinear because MPI communication, synchronization, and collective operation overhead become more significant at higher process counts. As each process receives fewer rows, the relative cost of `MPI_Allreduce`, `MPI_Reduce`, synchronization, and MPI runtime overhead becomes larger.

The project remains focused on demonstrating distributed-memory training, row distribution, class distribution, speedup, efficiency, and scalability for a parallel machine learning workload.

## Limitations

- The train/test split is deterministic and not shuffled.
- More robust evaluation could use random shuffling or k-fold cross-validation.
- The depth-4 tree is still intentionally shallow compared with production decision tree models.
- Only 15 evenly spaced thresholds per feature are tested.
- Wine quality labels are subjective and noisy.
- The project focuses on MPI parallelization, not maximum machine learning accuracy.

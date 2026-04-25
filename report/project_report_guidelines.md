# EC527 Final Project Report Guidelines

**Due Date:** 4/30 at 23:59

## Overview
The final write-up should incorporate feedback from your presentations. The write-up guidelines are essentially the same as the presentation guidelines, but tailored for a written format where the audience is the instructor, and it is expected to reach definite conclusions based on the project's outcomes.

## Required Content

Your report should cover the following points:

1. **Description of the Problem**
   - Provide a clear description of the application/problem you chose to optimize.

2. **Serial Code and Algorithm**
   - Describe what the serial code/algorithm looks like.
   - What is the specific algorithm?
   - What is the algorithmic complexity?

3. **Performance Profiling & Bottlenecks**
   - Where does the time go in the serial execution?
   - What is the arithmetic intensity of the workload?

4. **Data Structures & Memory**
   - What are the primary data structures used?
   - What is the memory reference pattern?

5. **Parallel Modifications**
   - Detail any modifications made to the algorithm to run in parallel (or justify the parallel algorithm selected if there were choices).

6. **Partitioning Strategy**
   - For the parallel (and GPU) parts, explain how the data and computations were partitioned.

7. **Optimizations Overview**
   - Provide an overview of your optimized codes.
   - What specific optimizations were implemented?
   - What problems or challenges did you encounter during optimization?

8. **Experiments and Results**
   - Present your experiments and results.
   - What worked well? Reach definite conclusions on the performance and scaling achieved.

## Grading Guidelines

The instructor will evaluate the report based on the following criteria:

* **Clarity of Write-up:** Is the report easy to follow? Are the assumptions reasonable and the logical progression sound?
* **Difficulty of Application:** Balances between application complexity and optimization depth. Straightforward applications require substantial and thorough optimizations.
* **Number of Architectures Tried:** One architecture per group member is OK, but more is better. For solo groups, depth on one architecture is acceptable, but exploring more is great.
* **Optimizations Attempted:** E.g., algorithm adjustments, memory mapping, pipeline accounting, vectorization, partitioning, synchronization, and parallelization methods (OpenMP, Pthreads, etc.). You don't need to do everything, but significant effort is expected.
* **New Stuff (Optional):** Extra credit/degree of difficulty for learning new paradigms (e.g., OpenCL, MPI) or targeting new architectures, though not strictly required.
* **Reference Codes and Validation:** High value placed on having an external reference code (like MKL, Matlab) used for performance validation and correctness checking.
* **Quality of Work:** 
  - What worked? What didn't? Why?
  - What limits did you hit?
  - How many ideas from the semester were applied, and how well does the report demonstrate an understanding of those ideas?

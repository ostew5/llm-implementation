# 3 Different Approaches to Implementing LLaMA 2's 15M Model

### Purpose

The purpose of this report is to discuss and compare 3 different approaches to implementing a LLaMA 2 15M model. The different approaches discussed are: 

- pure functional F# 
- impure function F#
- object-oriented C#

The report will primarily look into how efficient and effective each approach is by comparing readability, maintainability, conciseness and how fast the program runs.

This report assumes the reader has read and understands the attached project as well as previous knowledge of F#, C#, Python and C.

 ### Pure Functional F#

The Pure Functional F# approach uses traditional F# pipes to link functions together, however, this approach does reallocate memory every time a new memory size is required and for an easy development experience, F# does this memory allocation quite liberally. While prototyping LLM transformation algorithms in F#, this is useful, as it provides a "Python-like" development experience where Array's can be dynamically allocated to different dimensions allowing fast development without strict "C-like" memory management. However, for production, these dynamic allocations can be expensive on the CPU, especially when running on a single-thread. 

#### Pure F# Performance

The Pure F# implementation for LLaMA2 15M is implemented in the `functional/master` branch, and when running the transformer on Release Build in its default, 'tell a story,' configuration 3 times, this implementation provides the following results:

| Run (#) | Runtime (s) |
| ------- | ----------- |
| 1       | 5.854385    |
| 2       | 5.901534    |
| 3       | 5.904052    |

> The runtime is calculated by starting a `Stopwatch` from `System.Diagnostics` with `let stopwatch = Stopwatch.StartNew()` before `tellStory model53M mostLikely` and stopping it afterwards with `stopwatch.Stop()` and printing the elapsed total seconds. It doesn't account for loading the model or build time.
>
> `Main.fs`:
>
> ```F#
> open Transformer
> open IO
> open Decoders
> open System.Diagnostics
> 
> let model53M = readModel "model.bin" "tokenizer.bin"
> 
> let stopwatch = Stopwatch.StartNew()
> 
> tellStory model53M mostLikely
> 
> stopwatch.Stop()
> printfn "Elapsed time: %f seconds" (stopwatch.Elapsed.TotalSeconds)
> ```

#### Pure F# Performance (Multi-threading)

By simply replacing `Array.map` with `Array.Parallel.map` for all implemented functions the following results are achieved. (again with the same Release Build and `System.Diagnostic` tool)

| Run (#) | Runtime (s) |
| ------- | ----------- |
| 1       | 2.440932    |
| 2       | 2.386232    |
| 3       | 2.412346    |

### Impure Functional F#

The main drawback for the Pure F# implementation is the constant memory allocation and cleanup. To avoid this you can manually allocate a few arrays and reuse these where needed in loops that are commutatively intense. On top of this, the previous F# piping aren't used here to ensure proper handling of the mutable variables.

#### Impure F# Performance

After recreating most functions to accept mutable arguments and modifying the `feedfowardOneLayer` function to use these functions, the following results are achieved.

| Run (#) | Runtime (s) |
| ------- | ----------- |
| 1       | 4.671593    |
| 2       | 4.653250    |
| 3       | 4.665625    |

#### Impure F# Performance (Multi-threading)

By simply replacing `Array.map` with `Array.Parallel.map` and using `Parallel.For` instead of `for` loops where applicable, the following results are achieved:

| Run (#) | Runtime (s) |
| ------- | ----------- |
| 1       | 2.453104    |
| 2       | 2.358491    |
| 3       | 2.303801    |

### C# Object-Oriented

This C# Object-Oriented implementation of the LLM was provided in the assessment details and achieves the following results:

#### C# Object-Oriented Performance

Running the given C# example, the following results are achieved:

| Run (#) | Runtime (s) |
| ------- | ----------- |
| 1       | 2.169092    |
| 2       | 2.149198    |
| 3       | 2.164045    |

## Comparison

From the above tables, the mean runtimes for each implementation can be sorted from fastest to slowest:

| Implementation              | Mean Runtime (s) |
| --------------------------- | ---------------- |
| C# Object-Oriented          | 2.16             |
| Impure F# (Multi-threading) | 2.37             |
| Pure F# (Multi-threading)   | 2.41             |
| Impure F#                   | 4.66             |
| Pure F#                     | 5.89             |

From this list it can be seen that C# beats all, however from a code readability standpoint, the C# "Object-Oriented" approach has a lot of for loops that could be further segmented through inline functions that would improve readability and maintainability while not changing the generated machine code. 

For maintainability and readability, the Pure F# with Multi-threading combines a very easy to read code with a very fast runtime, only 300 milliseconds off the C# implementation.
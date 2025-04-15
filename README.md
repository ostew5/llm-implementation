# 3 Different Approaches to Implementing LLaMA 2's 15M Model

### Purpose

The purpose of this report is to discuss and compare 3 different approaches to implementing a LLaMA 2 15M model. The different approaches discussed are: 

- pure functional F# 
- impure function F#
- object-oriented C#

The report will primarily look into how efficient and effective each approach is by comparing readability, maintainability, conciseness and how fast the program runs.

This report assumes the reader has read and understands the attached project as well as previous knowledge of F#, C#, Python and C.

### Development Environment

For the purposes of comparing the efficiency of the code, the code is compiled and run on a computer and the program's runtime is used to compare the different implementations. For standardization, the specs of the computer this was run on is below:

| Item                            | Value                                                        |
| ------------------------------- | ------------------------------------------------------------ |
| System Model                    | Zenbook UM3402YA_UM3402YA                                    |
| OS Name                         | Microsoft Windows 11 Home                                    |
| Processor                       | AMD Ryzen 7 5825U with Radeon Graphics, 2000 Mhz, 8 Core(s), 16 Logical Processor(s) |
| Installed Physical Memory (RAM) | 16.0 GB                                                      |
| Battery Mode                    | Standard Mode - Plugged in                                   |

 ### Pure Functional F#

The Pure Functional F# approach uses traditional F# pipes to link functions together, however, this approach does reallocate memory every time a new memory size is required and for an easy development experience, F# does this memory allocation quite liberally. While prototyping LLM transformation algorithms in F#, this is useful, as it provides a "Python-like" development experience where Array's can be dynamically allocated to different dimensions allowing fast development without strict "C-like" memory management. However, for production, these dynamic allocations can be expensive on the CPU, especially when running on a single-thread. 

#### Pure F# Performance

The Pure F# implementation for LLaMA2 15M is implemented in the `functional/master` branch, and when running the transformer on Release Build in its default, 'tell a story,' configuration 3 times, this implementation provides the following results:

| Run (#) | Runtime (s) | RAM used (MiB) | CPU usage (%) |
| ------- | ----------- | -------------- | ------------- |
| 1       | 10.185      | 301.30         | 6.63          |
| 2       | 10.156      | 302.47         | 6.74          |
| 3       | 10.688      | 302.42         | 6.44          |


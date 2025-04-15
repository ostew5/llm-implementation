module HelperFunctions

open System.Diagnostics
open System.Numerics
open Types
open System.Threading.Tasks

// Vector operations ...

// Computes the product of x and y by multiplying the matrix w by the vector x.
// Given matrix w with size m x n, the input vector should be of length m, and
// the output vector size n.
let matrixMultiply (weights: WeightMatrix) (input: Vector) : Vector =
    //weights |> Array.Parallel.map (fun row -> Array.map2 (*) row input |> Array.sum)
    weights |> Array.Parallel.map (fun row -> Array.map2 (*) row input |> Array.sum)

// matrixMultiply that uses a mutable output
let mmatrixMultiply (output: Vector) (weights: WeightMatrix) (input: Vector) =
    Parallel.For(0, output.Length, fun i ->
        let mutable sum = 0.0
        for j in 0 .. input.Length - 1 do
            sum <- sum + weights.[i].[j] * input.[j]
        output.[i] <- sum
    ) |> ignore

// Adds two vectors together element-wise, returns a new vector.
// Both vectors should be of the dimension (array size).
let add (a: Vector) (b: Vector) : Vector =
    Array.map2 (+) a b

// add that uses a mutable output
let madd (output: Vector) (a: Vector) (b: Vector) =
    Parallel.For(0, output.Length, fun i ->
        output.[i] <- a.[i] + b.[i]
    ) |> ignore

// Multiplies two vectors together element-wise, returns a new vector.
// Both vectors should be of the dimension (array size).
let elementWiseMultiply (a : Vector) (b: Vector) : Vector =  
    Array.map2 (*) a b

// elementWiseMultiply that uses a mutable output
let melementWiseMultiply (output: Vector) (a: Vector) (b: Vector) =
    Parallel.For(0, output.Length, fun i ->
        output.[i] <- a.[i] * b.[i]
    ) |> ignore

// Performs root mean square (RMS) layer normalization on an input vector.
// To apply RMS layer normalization, we compute:
//   rms = sqrt(mean(input ^ 2))
//   output[i] = (output[i] / rms) * weights[i]
// Weights is a learnt parameter that is computed during training.
let rootMeanSquareNormalize (weights: WeightVector) (input: Vector) : Vector =
    let ss = 
        input
        |> Array.Parallel.map (fun x -> x * x)
        |> Array.average
        |> fun x -> x + 1e-5 // Avoid division by zero by adding a very small number (aka epsilon).
        |> fun x -> 1.0 / sqrt x // Compute rms & avoid a division in the loop

    (weights, input) ||> Array.map2 (fun w x -> w * (ss * x)) // Compute (output[i] / rms) * weights[i].

// Applies the softmax function to the given input vector (array).
// Softmax is a function that transforms a vector into a probability distribution,
// ranging from 0 to 1. Softmax is computed as:
//   softmax(xi) = exp(xi) / sum(exp(xj))
let softMax (input: Vector) : Vector =
    let max = input |> Array.max

    // Compute exp(x - max_val) for each vector element, and accumulate.
    let aggregatedInput = input |> Array.Parallel.map (fun x -> System.Math.Exp(x - max))
    let aggregatedSum = aggregatedInput |> Array.sum

    // Normalize by dividing by the sum, ensuring that all output values
    // are between 0 and 1.
    aggregatedInput |> Array.Parallel.map (fun x -> x / aggregatedSum)

// softMax that uses a mutable output
let msoftMax (output: Vector) (input: Vector) =
    let mutable max = input.[0]

    for i in 1 .. input.Length - 1 do
        if max < input.[i] then
            max <- input[i]

    let mutable sum = 0.0
    for i in 0 .. input.Length - 1 do
        output.[i] <- System.Math.Exp(input.[i] - max)
        sum <- sum + output.[i]

    Parallel.For(0, output.Length, fun i ->
        output.[i] <- output.[i] / sum
    ) |> ignore

// Applies our activation function: Sigmoid Linear Unit (SilU)
// SilU is computed as silu(x) = x*σ(x).
// σ(x) is the logistic sigmoid, or σ(x) = 1 / 1 + exp(-x). 
let sigmoidActivation (input:Vector) : Vector =
    input |> Array.Parallel.map (fun x -> x * (1.0 / (1.0 + System.Math.Exp(-x))))

// sigmoidActivation that uses mutable output
let msigmoidActivation (output: Vector) (input: Vector) =
    Parallel.For(0, output.Length, fun i ->
        output.[i] <- input.[i] * (1.0 / (1.0 + System.Math.Exp(-input.[i])))
    ) |> ignore

// Reshaping function ...

// To operate on each head of multi-head attention independently, each head
// needs its own sub-array. This function is responsible for dividing up the
// input vector into head-sized vectors, which are also arrays.
let reshapeToMultipleHeads (headSize: int)(input: Vector) : MultiHead = 
    Debug.Assert(input.Length % headSize = 0)
    input |> Array.chunkBySize headSize

// After computing multi-head attention, this function is responsible for
// recombining the separate head vectors into a single vector that forms
// the output for the layer.
let flattenMultipleHeads (input:MultiHead) : Vector =
    input |> Array.concat

// flattenMultipleHeads 
let mflattenMultipleHeads (output: Vector) (input: MultiHead) =
    let mutable index = 0
    for i = 0 to input.Length - 1 do
        let row = input.[i]
        for ii = 0 to row.Length - 1 do
            output.[index] <- row.[ii]
            index <- index + 1

// Converts the vector for each head into a series of two-dimensional positions
// suitable for Rotary Position Embedding (RoPE). The two dimensions are represented
// using a complex number type, with real and imaginary parts corresponding to the
// X and Y coordinates respectively.
let toComplex (vector: Vector) : Complex[] =
    Debug.Assert(vector.Length % 2 = 0)
    vector
    |> Array.chunkBySize 2
    |> Array.Parallel.map (fun [|real;imag|] -> Complex(real,imag))

// toComplex with a mutable output
let mtoComplex (output: Complex[]) (vector: Vector) =
    Debug.Assert(vector.Length % 2 = 0)
    let len = vector.Length / 2
    for i = 0 to len - 1 do
        let real = vector.[2 * i]
        let imag = vector.[2 * i + 1]
        output.[i] <- Complex(real, imag)

// Converts the list of 2D coordinates back to a single vector after applying
// Rotary Postion Embedding.
let flattenComplex (vector: Complex[]): Vector =    
    vector |> Array.collect (fun c -> [|c.Real; c.Imaginary|])

// flattenComplex with a mutable output
let mflattenComplex (output: Vector) (vector: Complex[]) =
    for i = 0 to vector.Length - 1 do
        let c = vector.[i]
        output.[2 * i]     <- c.Real
        output.[2 * i + 1] <- c.Imaginary

// Rotation functions ...

// Applies Rotary Position Embedding to a single pair of coordinates.
// i.e. for each pair of 2D coordinates, multiply by the corresponding rotationCoefficients.
let rotateOneHead (rotationCoeffients: Complex[]) (input: Complex[]) : Complex[] =
    (input, rotationCoeffients) ||> Array.map2 (fun left right -> Complex.Multiply(left, right))

// rotateOneHead with a mutable output
let mrotateOneHead (output: Complex[]) (rotationCoefficients: Complex[]) (input: Complex[]) =
    Parallel.For(0, output.Length, fun i ->
        output.[i] <- Complex.Multiply(input.[i], rotationCoefficients.[i])
    ) |> ignore

// Applies Rotary Position Embedding to each head of the input vector.
// You should use the utility functions above to convert the input vector into a series
// of 2D points, rotate them, and then merge it back to a single vector for each head.
let rotateVector (rotationCoeffients: Complex[]) (input: MultiHead) : MultiHead = 
    input 
        |> Array.Parallel.map toComplex
        |> Array.Parallel.map (fun x -> (rotationCoeffients, x) ||> rotateOneHead)
        |> Array.Parallel.map flattenComplex

// rotate Vector with a mutable output
let mrotateVector (output: MultiHead) (rotationCoefficients: Complex[]) (input: MultiHead) = 
    Parallel.For(0, input.Length, fun i ->
        let mutable temp: Complex[] = Array.zeroCreate (input.[i].Length / 2)
        mtoComplex temp input.[i]
        mrotateOneHead temp rotationCoefficients temp
        mflattenComplex output.[i] temp
    ) |> ignore

//HelperFunctionUnitTests
//  Tests in group: 100
//   Total Duration: 186 ms
//
//Outcomes
//   100 Passed
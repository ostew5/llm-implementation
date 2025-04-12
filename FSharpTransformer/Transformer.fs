module Transformer

open Types
open IO
open HelperFunctions
open Attention



// Return a new function that takes the head number, requested position, and position within head, returning the key/value
// from [head number][postion within head] in either the newValues matrix if the token position is the current position,
// otherwise in the history/cache table at [requested position][layer].
let createLookupFunction (previousValues:MultiHead[][]) (newValues:MultiHead) (tokenPosition:int) (layer:int): (int -> int -> int -> float) =
    fun headNumber requestedPosition positionWithinHead ->
        if tokenPosition = requestedPosition then
            newValues.[headNumber].[positionWithinHead]
        else
            previousValues.[requestedPosition].[layer].[headNumber].[positionWithinHead]
            

// Processes one layer of the transformer model. This is equivalent to the first for loop in the C# transformer() function.
// The parameters you will need are stored in the model.weights array under index layer.
// You need to:
// - Apply layer normalization to the current vector before attention using normalizeInputWeights
// - Generate query, key and value vectors by multiplying the current vector by the corresponding query (wq), key (qk) and value (wv)
//   matrices for this layer. You will need to use the reshapeToMultipleHeads() function to split these vectors.
// - Apply Rotary Position Embedding(RoPE) to query and key vectors. The value vector is not rotated.
// - Use the attention function to compute multi-head attention for the query/key/value vectors.
// - Project concatenated attention outputs with the output matrix (wo) to produce final attention.
// - Add the residual connection (input vector).
// - Apply layer normalization before the final feed-forward neural network (normalizeAttentionWeights).
// - Feed-forward network component: Matrix multiply w1 and w3, sigmoid is only applied to w1.
// - Then the product of these two matrices is multiplied by w2 with second residual connection.
let feedforwardOneLayer (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input: Vector) (layer: int) : Vector * MultiHead * MultiHead =
    let currentVector = 
        input |> rootMeanSquareNormalize model.weights.[layer].normalizeInputWeights

    let query = 
        matrixMultiply model.weights.[layer].wq currentVector
        |> fun x -> reshapeToMultipleHeads model.headSize x
        |> fun x -> rotateVector model.rotationCoefficients.[tokenPosition] x

    let key = 
        matrixMultiply model.weights.[layer].wk currentVector
        |> fun x -> reshapeToMultipleHeads model.headSize x
        |> fun x -> rotateVector model.rotationCoefficients.[tokenPosition] x

    let value = 
        matrixMultiply model.weights.[layer].wv currentVector
        |> fun x -> reshapeToMultipleHeads model.headSize x

    let keyLookup = createLookupFunction keyCache key tokenPosition layer
    let valueLookup = createLookupFunction valueCache value tokenPosition layer

    let residual = 
        attention keyLookup valueLookup tokenPosition query
        |> matrixMultiply model.weights.[layer].wo
        |> add input
    
    let normal =
        residual |> rootMeanSquareNormalize model.weights.[layer].normalizeAttentionWeights
    
    let hb = 
        matrixMultiply model.weights.[layer].w1 normal
        |> sigmoidActivation

    let hb2 = matrixMultiply model.weights.[layer].w3 normal

    ((hb, hb2) ||> elementWiseMultiply
        |> matrixMultiply model.weights.[layer].w2
        |> fun x -> add residual x, key, value)

// Returns a new array with the newElement added to array.
let appendElement (array: 'T[]) (newElement: 'T) : 'T[] =
    Array.append array [| newElement |]

// Feeds an input vector through all layers of the transformer.
// This function is also responsible for updating the key/value cache that is used to retrieve the vectors in later layers.
// The cache is "updated" for each layer by appending to the end of the array representing the cache.
let feedForwardAllLayers (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (input:Vector)  : Vector * MultiHead[] * MultiHead[] =
    // Use List.fold to process each layer, accumulating the input with each.
    let Folder (input, previousKeys, previousValues) layer =
        let (output, keys, values) = feedforwardOneLayer model keyCache valueCache tokenPosition input layer
        (output, appendElement previousKeys keys,  appendElement previousValues values)
    List.fold Folder (input, Array.empty, Array.empty) [0 .. model.numberOfLayers-1]

// Uses all the transformer model's layers to predict the next token that follows token.
// The output is the logits for each token, and the key/value cache for all layers for this token.
// This function roughly equates to the first copy() call and final rmsnorm()/matmul() calls in the C# transformer() method.
let feedForward (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][]) (tokenPosition:int) (token:Token) : Vector * MultiHead[] * MultiHead[] =
        let output, updatedKeyCache, updatedValueCache = 
            model.tokenEmbedding.[token] |> fun x -> feedForwardAllLayers model keyCache valueCache tokenPosition x

        (output
            |> rootMeanSquareNormalize model.normalizeOutputWeights
            |> matrixMultiply model.tokenEmbedding, updatedKeyCache, updatedValueCache)

// Obtains the logits for the next token, and selects the token to return based on the provided decoder function.
// You should also return the updated key/value cache.
let generateNextToken (model: Model) (keyCache:MultiHead[][]) (valueCache:MultiHead[][])  (tokenPosition:int) (token:Token) (decoder:Vector->Token) : Token * MultiHead[] * MultiHead[] =
    feedForward model keyCache valueCache tokenPosition token
        |||> fun logits updatedKeyCache updatedValueCache -> ((decoder logits), updatedKeyCache, updatedValueCache)

// Generates a sequence of tokens using the specified decoder.
// This function is responsible for appending the cache of key/values for all layers to the "main" key/value cache,
// which contains the key/values for all layers of every preceding token.
// The start and end of the sequence are indicated by the token 1, therefore we should stop producing tokens after
// a token of 1 is predicted. Each token is also printed out as it is generated.
let generateTokenSequence (model: Model) (decoder:Vector->Token) : string seq = 
    (1, 0, Array.empty, Array.empty) 
    |> Seq.unfold (fun (token, tokenPosition, previousKeys, previousValues) -> 
        let (nextToken, keys, values) = generateNextToken model previousKeys previousValues tokenPosition token decoder
        let newKeys = appendElement previousKeys keys
        let newValues = appendElement previousValues values
        if nextToken = 1 || tokenPosition+1 = model.seqenceLength
        then None
        else
            Some (model.vocabulary.[nextToken], (nextToken, tokenPosition+1, newKeys, newValues)))

let tellStory (model: Model) (decoder:Vector->Token) : unit =
    generateTokenSequence model decoder
    |> printStory

//TransformerTests
//  Tests in group: 13
//   Total Duration: 59.4 sec
//
//Outcomes
//   13 Passed

// master push

//  Tests in group: 151
//   Total Duration: 1.1 min
//
//Outcomes
//   151 Passed

// added extra transformer tests

//  Tests in group: 166
//   Total Duration: 1 min
//
//Outcomes
//   166 Passed


// Optimizations

// used Array.Parallel for all possible instances
//  Tests in group: 166
//   Total Duration: 25.1 sec
//
//Outcomes
//   166 Passed
// Result: 58.1% improvement on execution time

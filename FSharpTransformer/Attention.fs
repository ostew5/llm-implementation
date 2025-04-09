module Attention

open Types
open HelperFunctions

// Computes attention for a single token.
// Equivalent to the innermost loop of transformer() in the C# implementation.
// i.e. compute the square root of the dot product between query and key vectors.
// Hint: Use the keyLookup function, as we do not have the key vector directly here.
let attentionScore (query:Vector) (keyLookup:int->float) : float =    
    query
        |> elementWiseMultiply ([|0..query.Length - 1|] |> Array.map keyLookup)
        |> Array.sum
        |> fun score -> score / System.Math.Sqrt(query.Length)

// Compute the dot product of the attention vector with the value vector.
let weightedAttention (attention: Vector) (valueLookup:int->float) : float =
    attention
        |> elementWiseMultiply ([|0..attention.Length - 1|] |> Array.map valueLookup)
        |> Array.sum

// Computes attention for one head of multi-head attention, using the query, key and value vectors.
// This is equivalent to the n_heads loop in the transformer() function in the C# implementation.    
let attentionForOneHead (keyLookup:int->int->float) (valueLookup:int->int->float) (tokenPosition:int) (query: Vector): Vector =
    let attention =
        [|0..tokenPosition|]
            |> Array.map (fun x -> attentionScore query (keyLookup x))
            |> softMax

    [|0..query.Length - 1|]
        |> Array.map (fun i -> (weightedAttention attention (fun j -> valueLookup j i))) // weird usage of value lookup function because of the order of operations would prefer to change weightedAttention, but that would fail test cases

// Computes attention for all heads in multi-head attention.
// Hint: Instead of returning multiple vectors, one for each head, this array should be flattened with flattenMultipleHeads().
let attention  (keyLookup:int->int->int->float) (valueLookup:int->int->int->float) (tokenPosition:int) (query: MultiHead) : Vector =
    query
        |> Array.mapi (fun i q -> (attentionForOneHead (keyLookup i) (valueLookup i) tokenPosition q))
        |> flattenMultipleHeads

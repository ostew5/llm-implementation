open Transformer
open IO
open Decoders

let model53M = readModel "model.bin" "tokenizer.bin"
open System.Diagnostics

let stopwatch = Stopwatch.StartNew()
tellStory model53M mostLikely
stopwatch.Stop()
printfn "Elapsed time: %f seconds" (stopwatch.Elapsed.TotalSeconds)
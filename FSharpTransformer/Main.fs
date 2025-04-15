open Transformer
open IO
open Decoders
open System.Diagnostics

let model53M = readModel "model.bin" "tokenizer.bin"

let stopwatch = Stopwatch.StartNew()

tellStory model53M mostLikely

stopwatch.Stop()
printfn "Elapsed time: %f seconds" (stopwatch.Elapsed.TotalSeconds)
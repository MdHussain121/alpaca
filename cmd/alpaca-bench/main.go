// Command alpaca-bench runs throughput, latency, TTFT, and memory benchmarks across llama-server backends.
package main

import (
	"fmt"
	"os"

	"github.com/MdHussain121/alpaca/internal/bench"
)

func main() {
	if err := bench.Run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "alpaca-bench: %v\n", err)
		os.Exit(1)
	}
}

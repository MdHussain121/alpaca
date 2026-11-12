// Command eval runs the 10-task agent tool-use suite across real GGUF models.
package main

import (
	"fmt"
	"os"

	"github.com/MdHussain121/alpaca/internal/eval"
)

func main() {
	if err := eval.Run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "alpaca eval: %v\n", err)
		os.Exit(1)
	}
}

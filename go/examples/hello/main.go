// Command hello fetches loudr-1, speaks one line and writes hello.wav.
package main

import (
	"log"

	loudkit "github.com/loudreader/loudkit/go"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// run holds the body so the deferred Close runs on the way out: log.Fatal
// ends the process through os.Exit, which runs no deferred function.
func run() error {
	eng, err := loudkit.Load("loudreader/loudr-1")
	if err != nil {
		return err
	}
	defer eng.Close()

	v, err := eng.Voice("joe")
	if err != nil {
		return err
	}

	out, err := eng.Synthesize("Hello from loudkit.", v, loudkit.Options{Seed: 7})
	if err != nil {
		return err
	}
	if err := out.SaveWav("hello.wav"); err != nil {
		return err
	}
	log.Printf("hello.wav: %.2fs", out.Duration().Seconds())
	return nil
}

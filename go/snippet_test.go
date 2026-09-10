package loudkit

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The pages a stranger copies a Go block from, relative to the repository root.
var snippetPages = []string{"README.md", "site/src/handwritten/index.mdx"}

var goFence = regexp.MustCompile("(?s)```go\n(.*?)```")

// TestTheUserPageSnippetsVet pastes every ```go block on the front page and the
// landing page into a main and runs `go vet` on it against this module, so a
// block that does not compile fails here rather than in a reader's editor.
//
// The block is a fragment: it names `loudkit` and `log` without importing
// them, the way the README shows it, so the imports are supplied around it.
func TestTheUserPageSnippetsVet(t *testing.T) {
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	goMod, err := os.ReadFile(filepath.Join(root, "go", "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	goSum, err := os.ReadFile(filepath.Join(root, "go", "go.sum"))
	if err != nil {
		t.Fatal(err)
	}
	goLine := "go 1.25"
	for _, line := range strings.Split(string(goMod), "\n") {
		if strings.HasPrefix(line, "go ") {
			goLine = line
		}
	}
	for _, page := range snippetPages {
		text, err := os.ReadFile(filepath.Join(root, page))
		if err != nil {
			t.Fatal(err)
		}
		blocks := goFence.FindAllStringSubmatch(string(text), -1)
		if len(blocks) == 0 {
			t.Errorf("%s shows no go block; the gate is looking at nothing", page)
		}
		for i, m := range blocks {
			body := dedent(m[1])
			dir := t.TempDir()
			mod := "module snippet\n\n" + goLine + "\n\n" +
				"require github.com/loudreader/loudkit/go v0.0.0\n\n" +
				"replace github.com/loudreader/loudkit/go => " + filepath.Join(root, "go") + "\n"
			imports := "\tloudkit \"github.com/loudreader/loudkit/go\"\n"
			if strings.Contains(body, "voice.Load") {
				imports += "\t\"github.com/loudreader/loudkit/go/voice\"\n"
			}
			if strings.Contains(body, "log.") {
				imports = "\t\"log\"\n\n" + imports
			}
			main := "package main\n\nimport (\n" + imports + ")\n\nfunc main() {\n" + body + "}\n"
			for name, content := range map[string]string{"go.mod": mod, "go.sum": string(goSum), "main.go": main} {
				if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			cmd := exec.Command("go", "vet", ".")
			cmd.Dir = dir
			// The scratch module's graph is this module's graph, which `go test`
			// has already downloaded, so the proxy is off: the check never
			// reaches the network, and a cold cache fails here with a message.
			cmd.Env = append(os.Environ(), "GOFLAGS=-mod=mod", "GOWORK=off", "GOPROXY=off")
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Errorf("%s: go block %d does not vet:\n%s\n--- the block ---\n%s", page, i+1, out, body)
			}
		}
	}
}

// dedent strips the common leading indentation a page's fence sits at (the
// landing page indents its blocks inside a tab component).
func dedent(block string) string {
	lines := strings.Split(strings.TrimRight(block, "\n"), "\n")
	indent := -1
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		n := len(line) - len(strings.TrimLeft(line, " \t"))
		if indent < 0 || n < indent {
			indent = n
		}
	}
	if indent <= 0 {
		return block
	}
	for i, line := range lines {
		if len(line) >= indent {
			lines[i] = line[indent:]
		}
	}
	return strings.Join(lines, "\n") + "\n"
}

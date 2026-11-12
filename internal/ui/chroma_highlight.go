package ui

import (
	"strings"
	"sync"

	"github.com/alecthomas/chroma/v2"
)

type langSpec struct {
	name            string
	keywords        []string
	types           []string
	builtins        []string
	literals        []string
	lineComment     string
	hashComment     bool
	doubleDash      bool
	tripleQuotes    bool
	backtickStrings bool
	caseInsensitive bool
}

var (
	chromaMu         sync.Mutex
	chromaLexerCache = map[string]chroma.Lexer{}
	chromaStyleCache = map[string]*chroma.Style{}
)

// chromaLexerFor returns a lightweight Chroma RegexLexer without pulling in
// chroma/v2/lexers (whose 8+ MiB of embedded XML lexers bloat WorkingSetSize).
func chromaLexerFor(lang string) chroma.Lexer {
	key := normalizeLangKey(lang)
	chromaMu.Lock()
	defer chromaMu.Unlock()
	if l, ok := chromaLexerCache[key]; ok {
		return l
	}
	l := buildChromaLexer(key)
	chromaLexerCache[key] = l
	return l
}

// chromaStyleForTheme builds a Chroma Style harmonized with the active Launcher theme.
func chromaStyleForTheme(t Theme) *chroma.Style {
	name := t.Name
	if name == "" {
		name = "mocha"
	}
	chromaMu.Lock()
	defer chromaMu.Unlock()
	if s, ok := chromaStyleCache[name]; ok {
		return s
	}
	s := chroma.MustNewStyle("launcher-"+name, chroma.StyleEntries{
		chroma.Text:               string(t.Text),
		chroma.Whitespace:         string(t.Muted),
		chroma.Comment:            "italic " + string(t.Muted),
		chroma.CommentSingle:      "italic " + string(t.Muted),
		chroma.CommentMultiline:   "italic " + string(t.Muted),
		chroma.CommentPreproc:     string(t.Warning),
		chroma.Keyword:            "bold " + string(t.Accent),
		chroma.KeywordConstant:    string(t.Warning),
		chroma.KeywordDeclaration: "bold " + string(t.Accent),
		chroma.KeywordType:        string(t.Warning),
		chroma.Name:               string(t.Text),
		chroma.NameBuiltin:        string(t.Accent2),
		chroma.NameFunction:       string(t.Accent2),
		chroma.NameClass:          string(t.Warning),
		chroma.NameVariable:       string(t.Text),
		chroma.NameAttribute:      string(t.Warning),
		chroma.NameTag:            "bold " + string(t.Accent),
		chroma.LiteralString:      string(t.Success),
		chroma.LiteralNumber:      string(t.Warning),
		chroma.Operator:           string(t.Muted),
		chroma.Punctuation:        string(t.Muted),
		chroma.GenericHeading:     "bold " + string(t.Accent),
		chroma.GenericSubheading:  "bold " + string(t.Accent2),
		chroma.GenericInserted:    string(t.Success),
		chroma.GenericDeleted:     string(t.Error),
		chroma.GenericStrong:      "bold",
		chroma.GenericEmph:        "italic",
	})
	chromaStyleCache[name] = s
	return s
}

func normalizeLangKey(lang string) string {
	k := strings.ToLower(strings.TrimSpace(lang))
	switch k {
	case "golang":
		return "go"
	case "py", "python3":
		return "python"
	case "js", "jsx", "mjs", "cjs", "javascript":
		return "javascript"
	case "ts", "tsx", "mts", "typescript":
		return "typescript"
	case "rs":
		return "rust"
	case "c++", "cc", "cxx", "hpp", "hh":
		return "cpp"
	case "h":
		return "c"
	case "sh", "shell", "zsh":
		return "bash"
	case "ps1", "pwsh", "ps":
		return "powershell"
	case "jsonc", "jsonl":
		return "json"
	case "yml":
		return "yaml"
	case "md":
		return "markdown"
	case "patch":
		return "diff"
	case "htm", "xhtml", "svg", "xml":
		return "html"
	}
	return k
}

func buildChromaLexer(key string) chroma.Lexer {
	if key == "diff" {
		return chroma.Coalesce(chroma.MustNewLexer(
			&chroma.Config{Name: "diff", EnsureNL: true},
			func() chroma.Rules {
				return chroma.Rules{
					"root": {
						{Pattern: `^\+.*`, Type: chroma.GenericInserted, Mutator: nil},
						{Pattern: `^-.*`, Type: chroma.GenericDeleted, Mutator: nil},
						{Pattern: `^@@.*`, Type: chroma.GenericSubheading, Mutator: nil},
						{Pattern: `^(diff|index|---|\+\+\+).*`, Type: chroma.CommentPreproc, Mutator: nil},
						{Pattern: `.`, Type: chroma.Text, Mutator: nil},
					},
				}
			},
		))
	}
	if key == "markdown" {
		return chroma.Coalesce(chroma.MustNewLexer(
			&chroma.Config{Name: "markdown", EnsureNL: true},
			func() chroma.Rules {
				return chroma.Rules{
					"root": {
						{Pattern: `^#{1,6}\s+.*`, Type: chroma.GenericHeading, Mutator: nil},
						{Pattern: "`[^`]+`", Type: chroma.LiteralString, Mutator: nil},
						{Pattern: `\*\*[^*]+\*\*`, Type: chroma.GenericStrong, Mutator: nil},
						{Pattern: `\*[^*]+\*`, Type: chroma.GenericEmph, Mutator: nil},
						{Pattern: `^\s*[-*+]\s+`, Type: chroma.Keyword, Mutator: nil},
						{Pattern: `\[[^\]]+\]\([^)]+\)`, Type: chroma.NameBuiltin, Mutator: nil},
						{Pattern: `.`, Type: chroma.Text, Mutator: nil},
					},
				}
			},
		))
	}

	spec := specForLang(key)
	rules := func() chroma.Rules {
		root := chroma.Rules{}["root"]
		if spec.lineComment == "//" {
			root = append(root,
				chroma.Rule{Pattern: `//[^\n]*`, Type: chroma.CommentSingle},
				chroma.Rule{Pattern: `/\*.*?\*/`, Type: chroma.CommentMultiline},
			)
		}
		if spec.hashComment {
			root = append(root, chroma.Rule{Pattern: `#[^\n]*`, Type: chroma.CommentSingle})
		}
		if spec.doubleDash {
			root = append(root, chroma.Rule{Pattern: `--[^\n]*`, Type: chroma.CommentSingle})
		}
		if spec.tripleQuotes {
			root = append(root,
				chroma.Rule{Pattern: `"""[^"]*"""`, Type: chroma.LiteralString},
				chroma.Rule{Pattern: `'''[^']*'''`, Type: chroma.LiteralString},
			)
		}
		if spec.backtickStrings {
			root = append(root, chroma.Rule{Pattern: "`[^`]*`", Type: chroma.LiteralString})
		}
		if key == "html" {
			root = append(root,
				chroma.Rule{Pattern: `<!--.*?-->`, Type: chroma.CommentMultiline},
				chroma.Rule{Pattern: `</?[A-Za-z_][A-Za-z0-9:_-]*`, Type: chroma.NameTag},
				chroma.Rule{Pattern: `/?>`, Type: chroma.NameTag},
			)
		}
		root = append(root,
			chroma.Rule{Pattern: `"(?:\\.|[^"\\])*"`, Type: chroma.LiteralString},
			chroma.Rule{Pattern: `'(?:\\.|[^'\\])*'`, Type: chroma.LiteralString},
		)
		if len(spec.keywords) > 0 {
			root = append(root, chroma.Rule{Pattern: chroma.Words(`\b`, `\b`, spec.keywords...), Type: chroma.Keyword})
		}
		if len(spec.types) > 0 {
			root = append(root, chroma.Rule{Pattern: chroma.Words(`\b`, `\b`, spec.types...), Type: chroma.KeywordType})
		}
		if len(spec.builtins) > 0 {
			root = append(root, chroma.Rule{Pattern: chroma.Words(`\b`, `\b`, spec.builtins...), Type: chroma.NameBuiltin})
		}
		if len(spec.literals) > 0 {
			root = append(root, chroma.Rule{Pattern: chroma.Words(`\b`, `\b`, spec.literals...), Type: chroma.KeywordConstant})
		}
		root = append(root,
			chroma.Rule{Pattern: `\b0[xXoObB][0-9a-fA-F_]+\b`, Type: chroma.LiteralNumber},
			chroma.Rule{Pattern: `\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b`, Type: chroma.LiteralNumber},
			chroma.Rule{Pattern: `[A-Za-z_][A-Za-z0-9_]*(?=\s*\()`, Type: chroma.NameFunction},
			chroma.Rule{Pattern: `[A-Za-z_][A-Za-z0-9_]*`, Type: chroma.Name},
			chroma.Rule{Pattern: `[+\-*/%=!<>&|^~?:]+`, Type: chroma.Operator},
			chroma.Rule{Pattern: `[\[\]{}(),.;]`, Type: chroma.Punctuation},
			chroma.Rule{Pattern: `\s+`, Type: chroma.Whitespace},
			chroma.Rule{Pattern: `.`, Type: chroma.Text},
		)
		return chroma.Rules{"root": root}
	}

	return chroma.Coalesce(chroma.MustNewLexer(
		&chroma.Config{
			Name:            spec.name,
			CaseInsensitive: spec.caseInsensitive,
			EnsureNL:        true,
		},
		rules,
	))
}

func specForLang(key string) langSpec {
	switch key {
	case "go":
		return langSpec{
			name:            "go",
			keywords:        []string{"break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough", "for", "func", "go", "goto", "if", "import", "interface", "map", "package", "range", "return", "select", "struct", "switch", "type", "var"},
			types:           []string{"any", "bool", "byte", "comparable", "complex64", "complex128", "error", "float32", "float64", "int", "int8", "int16", "int32", "int64", "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64", "uintptr"},
			builtins:        []string{"append", "cap", "clear", "close", "complex", "copy", "delete", "imag", "len", "make", "max", "min", "new", "panic", "print", "println", "real", "recover"},
			literals:        []string{"true", "false", "nil", "iota"},
			lineComment:     "//",
			backtickStrings: true,
		}
	case "python":
		return langSpec{
			name:         "python",
			keywords:     []string{"and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "match", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"},
			types:        []string{"bool", "bytes", "dict", "float", "frozenset", "int", "list", "object", "set", "str", "tuple"},
			builtins:     []string{"abs", "all", "any", "enumerate", "filter", "isinstance", "len", "map", "max", "min", "open", "print", "range", "sorted", "sum", "type", "zip"},
			literals:     []string{"True", "False", "None"},
			hashComment:  true,
			tripleQuotes: true,
		}
	case "javascript", "typescript":
		return langSpec{
			name:            key,
			keywords:        []string{"as", "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "enum", "export", "extends", "finally", "for", "from", "function", "if", "implements", "import", "in", "instanceof", "interface", "let", "new", "of", "package", "private", "protected", "public", "return", "static", "super", "switch", "this", "throw", "try", "type", "typeof", "var", "void", "while", "with", "yield"},
			types:           []string{"any", "bigint", "boolean", "never", "number", "object", "string", "symbol", "unknown", "Array", "Map", "Promise", "Record", "Set"},
			builtins:        []string{"console", "Math", "JSON", "Object", "parseInt", "parseFloat", "setTimeout", "fetch"},
			literals:        []string{"true", "false", "null", "undefined", "NaN", "Infinity"},
			lineComment:     "//",
			backtickStrings: true,
		}
	case "rust":
		return langSpec{
			name:        "rust",
			keywords:    []string{"as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum", "extern", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "type", "unsafe", "use", "where", "while"},
			types:       []string{"bool", "char", "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "str", "String", "Vec", "Option", "Result", "Box", "u8", "u16", "u32", "u64", "u128", "usize"},
			builtins:    []string{"println", "format", "vec", "panic", "assert", "Some", "None", "Ok", "Err"},
			literals:    []string{"true", "false"},
			lineComment: "//",
		}
	case "c", "cpp":
		return langSpec{
			name:        key,
			keywords:    []string{"auto", "break", "case", "class", "const", "constexpr", "continue", "default", "delete", "do", "else", "enum", "explicit", "export", "extern", "for", "friend", "goto", "if", "inline", "namespace", "new", "noexcept", "operator", "private", "protected", "public", "return", "sizeof", "static", "struct", "switch", "template", "this", "throw", "try", "typedef", "typename", "union", "using", "virtual", "volatile", "while"},
			types:       []string{"bool", "char", "double", "float", "int", "long", "short", "signed", "size_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int64_t", "std", "string", "vector", "void"},
			literals:    []string{"true", "false", "nullptr", "NULL"},
			lineComment: "//",
			hashComment: true,
		}
	case "bash", "powershell":
		return langSpec{
			name:            key,
			keywords:        []string{"if", "then", "elif", "else", "fi", "for", "in", "do", "done", "while", "until", "case", "esac", "function", "return", "exit", "export", "local", "param", "foreach", "switch", "try", "catch", "finally"},
			builtins:        []string{"echo", "printf", "cd", "pwd", "ls", "cat", "grep", "rg", "sed", "awk", "curl", "git", "go", "Write-Host", "Get-ChildItem", "Set-Location"},
			literals:        []string{"true", "false", "$true", "$false", "$null"},
			hashComment:     true,
			backtickStrings: key == "bash",
		}
	case "json":
		return langSpec{
			name:     "json",
			literals: []string{"true", "false", "null"},
		}
	case "yaml", "toml", "ini":
		return langSpec{
			name:        key,
			literals:    []string{"true", "false", "null", "yes", "no"},
			hashComment: true,
		}
	case "sql":
		return langSpec{
			name:            "sql",
			keywords:        []string{"SELECT", "FROM", "WHERE", "INSERT", "INTO", "VALUES", "UPDATE", "SET", "DELETE", "CREATE", "TABLE", "INDEX", "DROP", "ALTER", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "GROUP", "BY", "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "AND", "OR", "NOT", "IN", "AS", "DISTINCT", "HAVING", "UNION", "PRIMARY", "KEY"},
			types:           []string{"INT", "INTEGER", "BIGINT", "TEXT", "VARCHAR", "BOOLEAN", "FLOAT", "REAL", "TIMESTAMP", "JSON", "JSONB"},
			builtins:        []string{"COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE"},
			literals:        []string{"TRUE", "FALSE", "NULL"},
			doubleDash:      true,
			caseInsensitive: true,
		}
	case "html", "css":
		return langSpec{
			name:        key,
			lineComment: "//",
		}
	default:
		name := key
		if name == "" {
			name = "text"
		}
		return langSpec{
			name:            name,
			keywords:        []string{"if", "else", "for", "while", "return", "func", "function", "def", "class", "struct", "type", "import", "from", "const", "let", "var"},
			literals:        []string{"true", "false", "null", "nil", "none", "None"},
			lineComment:     "//",
			hashComment:     true,
			backtickStrings: true,
		}
	}
}

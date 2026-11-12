package ui

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/MdHussain121/alpaca/internal/config"
	"github.com/MdHussain121/alpaca/internal/estimate"
	"github.com/MdHussain121/alpaca/internal/flags"
)

// field is one row in the settings panel. Every field mutates the real
// Settings struct, so the command preview and the launched command agree.
//
// A field with apply() can also be typed into: Enter opens the editor with
// edit() as its contents, and apply() parses whatever the user submits.
type field struct {
	label    string
	value    func() string
	slider   func() (float64, bool)
	delta    func(dir int)
	activate func()
	hidden   func() bool
	note     func() string
	edit     func() string
	apply    func(string) error
}

var (
	batchStops     = []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
	batchStepStops = []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}
	ubatchStops    = []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}
	maxTokenStops  = []int{0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}
)

func (m *Model) fields() []field {
	layerCount := m.layers()
	if layerCount == 0 {
		layerCount = 1
	}
	isMoE := m.meta != nil && m.meta.IsMoE()

	trained := int64(0)
	if m.meta != nil {
		trained = int64(m.meta.ContextLength)
	}

	fs := []field{
		{
			label:  "Context",
			value:  func() string { return formatTokensK(m.settings.Context) },
			slider: func() (float64, bool) { return stopRatio(m.settings.Context, flags.ContextStops()), true },
			delta: func(dir int) {
				m.settings.Context = flags.NextStop(m.settings.Context, dir)
			},
			edit: func() string { return strconv.Itoa(m.settings.Context) },
			apply: func(v string) error {
				n, err := parseInt(v, 256, 4*1024*1024)
				if err != nil {
					return fmt.Errorf("context: %v (tokens, 256..4194304)", err)
				}
				m.settings.Context = n
				return nil
			},
			note: func() string {
				if trained <= 0 {
					return ""
				}
				switch {
				case int64(m.settings.Context) > trained:
					return fmt.Sprintf("above trained %s", formatTokensK(int(trained)))
				case int64(m.settings.Context) == trained:
					return fmt.Sprintf("trained %s", formatTokensK(int(trained)))
				}
				return ""
			},
		},
		{
			label: "K cache",
			value: func() string {
				if m.settings.AutoKV && m.meta != nil {
					return m.resolved.KCache
				}
				return m.settings.KCache
			},
			delta: func(dir int) {
				if m.settings.AutoKV && m.meta != nil {
					m.settings.KCache = m.resolved.KCache
					m.settings.VCache = m.resolved.VCache
				}
				m.settings.AutoKV = false
				m.settings.KCache = cycle(flags.CacheOptions(), m.settings.KCache, dir)
			},
			note: func() string {
				if m.est.KVQualityNote != "" {
					return m.est.KVQualityNote
				}
				if m.settings.AutoKV && m.est.KVReason != "" {
					return m.est.KVReason
				}
				return ""
			},
		},
		{
			label: "V cache",
			value: func() string {
				if m.settings.AutoKV && m.meta != nil {
					return m.resolved.VCache
				}
				return m.settings.VCache
			},
			delta: func(dir int) {
				if m.settings.AutoKV && m.meta != nil {
					m.settings.KCache = m.resolved.KCache
					m.settings.VCache = m.resolved.VCache
				}
				m.settings.AutoKV = false
				m.settings.VCache = cycle(flags.CacheOptions(), m.settings.VCache, dir)
			},
			note: func() string {
				eff := m.settings
				if m.settings.AutoKV && m.meta != nil {
					eff = m.resolved
				}
				if eff.VCache == "q4_0" {
					return "forces fa · q4_0 trades recall"
				}
				if eff.FlashAttnLocked() {
					return "forces flash attn"
				}
				return ""
			},
		},
		{
			label: "Flash attn",
			value: func() string {
				eff := m.settings
				if m.settings.AutoKV && m.meta != nil {
					eff = m.resolved
				}
				if eff.FlashAttnLocked() {
					return "ON (required)"
				}
				return onOffLabel(eff.FlashAttn)
			},
			delta: func(dir int) {
				eff := m.settings
				if m.settings.AutoKV && m.meta != nil {
					eff = m.resolved
				}
				if eff.FlashAttnLocked() {
					return
				}
				m.settings.FlashAttn = !m.settings.FlashAttn
			},
			activate: func() {
				eff := m.settings
				if m.settings.AutoKV && m.meta != nil {
					eff = m.resolved
				}
				if eff.FlashAttnLocked() {
					return
				}
				m.settings.FlashAttn = !m.settings.FlashAttn
			},
		},
		{
			label: "GPU layers",
			value: func() string {
				if m.settings.GPULayers >= layerCount {
					return fmt.Sprintf("all (%d)", layerCount)
				}
				return fmt.Sprintf("%d/%d", m.settings.GPULayers, layerCount)
			},
			slider: func() (float64, bool) {
				cur := clampInt(m.settings.GPULayers, 0, layerCount)
				return float64(cur) / float64(layerCount), true
			},
			delta: func(dir int) {
				if dir < 0 && m.settings.GPULayers >= layerCount {
					next := layerCount - 1
					if next < 0 {
						next = 0
					}
					m.settings.GPULayers = next
					return
				}
				next := m.settings.GPULayers + dir
				if next < 0 {
					next = 0
				}
				if next >= layerCount {
					next = flags.AllCount
				}
				m.settings.GPULayers = next
			},
			edit: func() string {
				if m.settings.GPULayers >= layerCount {
					return "all"
				}
				return strconv.Itoa(m.settings.GPULayers)
			},
			apply: func(v string) error {
				t := strings.ToLower(strings.TrimSpace(v))
				if t == "all" || t == "max" || t == "-1" {
					m.settings.GPULayers = flags.AllCount
					return nil
				}
				n, err := parseInt(v, 0, layerCount)
				if err != nil {
					return fmt.Errorf("gpu layers: 0..%d, or \"all\"", layerCount)
				}
				if n >= layerCount {
					m.settings.GPULayers = flags.AllCount
				} else {
					m.settings.GPULayers = n
				}
				return nil
			},
			note: func() string {
				if m.settings.GPULayers == 0 {
					return "CPU only (model in RAM)"
				}
				if m.settings.GPULayers < layerCount {
					return fmt.Sprintf("%dL offloaded to CPU RAM", layerCount-m.settings.GPULayers)
				}
				return ""
			},
		},
		{
			label: "KV offload",
			value: func() string { return onOffLabel(!m.settings.NoKVOffload) },
			delta: func(dir int) { m.settings.NoKVOffload = !m.settings.NoKVOffload },
			activate: func() {
				m.settings.NoKVOffload = !m.settings.NoKVOffload
			},
			note: func() string {
				if m.settings.NoKVOffload {
					return "KV kept in host RAM (-nkvo)"
				}
				return "KV stored in VRAM"
			},
		},
		{
			label:  "MoE experts on CPU",
			hidden: func() bool { return !isMoE },
			value: func() string {
				if m.settings.NCpuMoe <= 0 {
					return "0 (GPU)"
				}
				return fmt.Sprintf("%d/%d", m.settings.NCpuMoe, layerCount)
			},
			slider: func() (float64, bool) {
				cur := clampInt(m.settings.NCpuMoe, 0, layerCount)
				return float64(cur) / float64(layerCount), true
			},
			delta: func(dir int) {
				m.settings.NCpuMoe = clampInt(m.settings.NCpuMoe+dir, 0, layerCount)
			},
			edit: func() string { return strconv.Itoa(m.settings.NCpuMoe) },
			apply: func(v string) error {
				n, err := parseInt(v, 0, layerCount)
				if err != nil {
					return fmt.Errorf("MoE layers on CPU: 0..%d", layerCount)
				}
				m.settings.NCpuMoe = n
				return nil
			},
			note: func() string {
				if m.settings.NCpuMoe == 0 {
					return ""
				}
				return "experts in RAM"
			},
		},
		{
			label: "MoE cache",
			hidden: func() bool {
				return !isMoE || !estimate.MoECacheAllowed(m.meta, m.est, m.hw, m.settings.MoeCacheMiB)
			},
			value: func() string {
				if m.settings.MoeCacheMiB <= 0 {
					return "off"
				}
				return fmt.Sprintf("%dM", m.settings.MoeCacheMiB)
			},
			delta: func(dir int) {
				capMiB := m.est.MoeCacheCapMiB
				if capMiB < 0 {
					capMiB = 0
				}
				cur := max(0, m.settings.MoeCacheMiB) + dir*256
				if cur < 0 {
					cur = 0
				} else if cur > capMiB {
					cur = capMiB
				}
				m.settings.MoeCacheMiB = cur
			},
			edit: func() string {
				if m.settings.MoeCacheMiB <= 0 {
					return "0"
				}
				return strconv.Itoa(m.settings.MoeCacheMiB)
			},
			apply: func(v string) error {
				t := strings.ToLower(strings.TrimSpace(v))
				if t == "off" || t == "" {
					m.settings.MoeCacheMiB = 0
					return nil
				}
				n, err := parseInt(v, 0, 65536)
				if err != nil {
					return fmt.Errorf("MoE cache: off or 0..65536 MiB")
				}
				if m.est.MoeCacheCapMiB > 0 && n > m.est.MoeCacheCapMiB {
					n = m.est.MoeCacheCapMiB
				}
				m.settings.MoeCacheMiB = n
				return nil
			},
			note: func() string {
				if m.est.MoeCacheReason != "" {
					return m.est.MoeCacheReason
				}
				return "manual opt-in (VRAM + RAM capped)"
			},
		},
		{
			label: "Auto fit",
			value: func() string { return onOffLabel(m.settings.AutoFit) },
			delta: func(dir int) {
				m.settings.AutoFit = !m.settings.AutoFit
				m.userEditedAutoFit = true
			},
			activate: func() {
				m.settings.AutoFit = !m.settings.AutoFit
				m.userEditedAutoFit = true
			},
		},
		{
			label: "Auto KV",
			value: func() string {
				if m.settings.AutoKV && m.meta != nil {
					return fmt.Sprintf("on (%s/%s)", m.resolved.KCache, m.resolved.VCache)
				}
				return onOffLabel(m.settings.AutoKV)
			},
			delta: func(dir int) {
				if m.settings.AutoKV && m.meta != nil {
					m.settings.KCache = m.resolved.KCache
					m.settings.VCache = m.resolved.VCache
				}
				m.settings.AutoKV = !m.settings.AutoKV
			},
			activate: func() {
				if m.settings.AutoKV && m.meta != nil {
					m.settings.KCache = m.resolved.KCache
					m.settings.VCache = m.resolved.VCache
				}
				m.settings.AutoKV = !m.settings.AutoKV
			},
			note: func() string {
				if m.est.KVQualityNote != "" {
					return m.est.KVQualityNote
				}
				if m.est.KVReason != "" {
					return m.est.KVReason
				}
				return "f16 > q8/q8 > q8/q4 > q4/q4 (<=16K)"
			},
		},
		{
			label: "Load mode",
			value: func() string {
				if m.settings.LoadMode == "auto" && m.meta != nil {
					return fmt.Sprintf("auto (%s)", estimate.SmartLoadMode(m.est, m.hw))
				}
				return m.settings.LoadMode
			},
			delta: func(dir int) {
				m.settings.LoadMode = cycle(flags.LoadModeOptions(), m.settings.LoadMode, dir)
			},
			note: func() string {
				if m.est.LoadModeWarning != "" {
					return m.est.LoadModeWarning
				}
				switch m.settings.LoadMode {
				case "auto":
					return "none on safe GPU fit, mmap otherwise"
				case "none":
					return "full read, needs RAM"
				case "mlock":
					return "pins in RAM"
				case "dio":
					return "bypasses cache"
				}
				return ""
			},
		},
		{
			label:    "Warmup",
			value:    func() string { return onOffLabel(m.settings.Warmup) },
			delta:    func(dir int) { m.settings.Warmup = !m.settings.Warmup },
			activate: func() { m.settings.Warmup = !m.settings.Warmup },
			note:     func() string { return "off skips startup pass (--no-warmup)" },
		},
		{
			label:    "Speculative",
			value:    func() string { return m.settings.SpecType },
			delta:    func(dir int) { m.settings.SpecType = cycle(flags.SpecTypeOptions(), m.settings.SpecType, dir) },
			activate: func() { m.settings.SpecType = cycle(flags.SpecTypeOptions(), m.settings.SpecType, 1) },
			note: func() string {
				if m.settings.SpecType == "ngram-mod" {
					return "speeds repetitive code edits"
				}
				return "opt-in draftless ngram-mod"
			},
		},
		{
			label: "Threads",
			value: func() string {
				if m.settings.Threads == 0 {
					if m.meta != nil && m.resolved.Threads > 0 {
						return fmt.Sprintf("auto (%d)", m.resolved.Threads)
					}
					return "auto (default)"
				}
				return fmt.Sprintf("%d", m.settings.Threads)
			},
			slider: func() (float64, bool) {
				maxT := m.hw.LogicCores
				if maxT < 16 {
					maxT = 16
				}
				cur := m.settings.Threads
				if cur == 0 && m.meta != nil {
					cur = m.resolved.Threads
				}
				return rangeRatio(float64(cur), 0, float64(maxT)), true
			},
			delta: func(dir int) {
				if m.settings.Threads == 0 && dir < 0 {
					return
				}
				m.settings.Threads = clampInt(m.settings.Threads+dir, 0, 256)
			},
			edit: func() string {
				if m.settings.Threads == 0 {
					return "auto"
				}
				return strconv.Itoa(m.settings.Threads)
			},
			apply: func(v string) error {
				t := strings.ToLower(strings.TrimSpace(v))
				if t == "auto" || t == "0" || t == "default" {
					m.settings.Threads = 0
					return nil
				}
				n, err := parseInt(v, 1, 256)
				if err != nil {
					return fmt.Errorf("threads: auto or 1..256")
				}
				m.settings.Threads = n
				return nil
			},
			note: func() string {
				if m.settings.Threads == 0 && m.est.ThreadsReason != "" {
					return m.est.ThreadsReason
				}
				if m.hw.PCores > 0 && m.hw.ECores > 0 {
					return fmt.Sprintf("%d phys (%dP+%dE)", m.hw.PhysCores, m.hw.PCores, m.hw.ECores)
				}
				if m.hw.PhysCores > 0 {
					return fmt.Sprintf("%d phys", m.hw.PhysCores)
				}
				return ""
			},
		},
		{
			label: "Priority",
			value: func() string {
				if m.settings.Prio == "auto" && m.meta != nil && m.resolved.Prio != "" {
					return fmt.Sprintf("auto (%s)", m.resolved.Prio)
				}
				return m.settings.Prio
			},
			delta: func(dir int) { m.settings.Prio = cycle(flags.PrioOptions(), m.settings.Prio, dir) },
			note: func() string {
				if m.settings.Prio == "auto" {
					return "high (--prio 2) on CPU dense"
				}
				return "process priority (--prio)"
			},
		},
		{
			label:  "Batch",
			value:  func() string { return fmt.Sprintf("%d", m.settings.Batch) },
			slider: func() (float64, bool) { return stopRatio(m.settings.Batch, batchStops), true },
			delta: func(dir int) {
				m.settings.Batch = nextIntStop(m.settings.Batch, dir, batchStepStops)
				if m.settings.UBatch > m.settings.Batch {
					m.settings.UBatch = m.settings.Batch
				}
			},
			edit: func() string { return strconv.Itoa(m.settings.Batch) },
			apply: func(v string) error {
				n, err := parseInt(v, 32, 65536)
				if err != nil {
					return fmt.Errorf("batch: 32..65536")
				}
				m.settings.Batch = n
				if m.settings.UBatch > n {
					m.settings.UBatch = n
				}
				return nil
			},
		},
		{
			label: "Ubatch",
			value: func() string { return fmt.Sprintf("%d", m.settings.UBatch) },
			slider: func() (float64, bool) {
				lo := min(64, m.settings.Batch)
				return rangeRatio(float64(m.settings.UBatch), float64(lo), float64(m.settings.Batch)), true
			},
			delta: func(dir int) {
				if m.settings.Batch <= 64 {
					m.settings.UBatch = m.settings.Batch
					return
				}
				var valid []int
				for _, s := range ubatchStops {
					if s <= m.settings.Batch {
						valid = append(valid, s)
					}
				}
				if len(valid) == 0 {
					m.settings.UBatch = m.settings.Batch
					return
				}
				if valid[len(valid)-1] < m.settings.Batch {
					valid = append(valid, m.settings.Batch)
				}
				m.settings.UBatch = nextIntStop(m.settings.UBatch, dir, valid)
			},
			edit: func() string { return strconv.Itoa(m.settings.UBatch) },
			apply: func(v string) error {
				n, err := parseInt(v, 1, m.settings.Batch)
				if err != nil {
					return fmt.Errorf("ubatch: 1..%d (batch)", m.settings.Batch)
				}
				m.settings.UBatch = n
				return nil
			},
		},
		{
			label:  "Parallel slots",
			value:  func() string { return fmt.Sprintf("%d", m.settings.Parallel) },
			slider: func() (float64, bool) { return rangeRatio(float64(m.settings.Parallel), 1, 16), true },
			delta:  func(dir int) { m.settings.Parallel = clampInt(m.settings.Parallel+dir, 1, 32) },
			edit:   func() string { return strconv.Itoa(m.settings.Parallel) },
			apply: func(v string) error {
				n, err := parseInt(v, 1, 32)
				if err != nil {
					return fmt.Errorf("parallel slots: 1..32")
				}
				m.settings.Parallel = n
				return nil
			},
		},
		{
			label:  "Temp",
			value:  func() string { return fmt.Sprintf("%.2f", m.settings.Temp) },
			slider: func() (float64, bool) { return rangeRatio(m.settings.Temp, 0, 2.0), true },
			delta: func(dir int) {
				m.settings.Temp = clampFloat(round2(m.settings.Temp+float64(dir)*0.05), 0, 5)
			},
			edit: func() string { return strconv.FormatFloat(m.settings.Temp, 'f', -1, 64) },
			apply: func(v string) error {
				f, err := parseFloat(v, 0, 5)
				if err != nil {
					return fmt.Errorf("temp: 0..5")
				}
				m.settings.Temp = f
				return nil
			},
		},
		{
			label:  "Top-p",
			value:  func() string { return fmt.Sprintf("%.2f", m.settings.TopP) },
			slider: func() (float64, bool) { return rangeRatio(m.settings.TopP, 0, 1.0), true },
			delta: func(dir int) {
				m.settings.TopP = clampFloat(round2(m.settings.TopP+float64(dir)*0.05), 0, 1)
			},
			edit: func() string { return strconv.FormatFloat(m.settings.TopP, 'f', -1, 64) },
			apply: func(v string) error {
				f, err := parseFloat(v, 0, 1)
				if err != nil {
					return fmt.Errorf("top-p: 0..1")
				}
				m.settings.TopP = f
				return nil
			},
		},
		{
			label:  "Top-k",
			value:  func() string { return fmt.Sprintf("%d", m.settings.TopK) },
			slider: func() (float64, bool) { return rangeRatio(float64(m.settings.TopK), 0, 100), true },
			delta:  func(dir int) { m.settings.TopK = clampInt(m.settings.TopK+dir*5, 0, 1000) },
			edit:   func() string { return strconv.Itoa(m.settings.TopK) },
			apply: func(v string) error {
				n, err := parseInt(v, 0, 1000)
				if err != nil {
					return fmt.Errorf("top-k: 0..1000")
				}
				m.settings.TopK = n
				return nil
			},
		},
		{
			label:  "Min-p",
			value:  func() string { return fmt.Sprintf("%.2f", m.settings.MinP) },
			slider: func() (float64, bool) { return rangeRatio(m.settings.MinP, 0, 1.0), true },
			delta: func(dir int) {
				m.settings.MinP = clampFloat(round2(m.settings.MinP+float64(dir)*0.05), 0, 1)
			},
			edit: func() string { return strconv.FormatFloat(m.settings.MinP, 'f', -1, 64) },
			apply: func(v string) error {
				f, err := parseFloat(v, 0, 1)
				if err != nil {
					return fmt.Errorf("min-p: 0..1")
				}
				m.settings.MinP = f
				return nil
			},
		},
		{
			label: "Reasoning",
			value: func() string { return m.settings.Reasoning },
			delta: func(dir int) {
				m.settings.Reasoning = cycle(flags.ReasoningOptions(), m.settings.Reasoning, dir)
			},
			note: func() string {
				if m.settings.Reasoning == "off" {
					return "thinking disabled"
				}
				return "thinking mode"
			},
		},
		{
			label: "Reasoning effort",
			value: func() string { return m.settings.ReasoningEffort },
			slider: func() (float64, bool) {
				return listRatio(m.settings.ReasoningEffort, flags.ReasoningEffortOptions()), true
			},
			delta: func(dir int) {
				m.settings.ReasoningEffort = cycle(flags.ReasoningEffortOptions(), m.settings.ReasoningEffort, dir)
			},
			note: func() string {
				if m.settings.Reasoning == "off" {
					return "ignored while off"
				}
				return ""
			},
		},
		{
			label: "Reasoning budget",
			value: func() string { return m.settings.ReasoningBudget },
			slider: func() (float64, bool) {
				return listRatio(m.settings.ReasoningBudget, flags.ReasoningBudgetOptions()), true
			},
			delta: func(dir int) {
				m.settings.ReasoningBudget = stepStringList(flags.ReasoningBudgetOptions(), m.settings.ReasoningBudget, dir)
			},
			edit: func() string { return m.settings.ReasoningBudget },
			apply: func(v string) error {
				t := strings.ToLower(strings.TrimSpace(v))
				if t == "" || t == "auto" || t == "-1" {
					m.settings.ReasoningBudget = "auto"
					return nil
				}
				n, err := parseInt(t, 0, 1048576)
				if err != nil {
					return fmt.Errorf("budget: \"auto\" or 0..1048576 tokens")
				}
				m.settings.ReasoningBudget = strconv.Itoa(n)
				return nil
			},
		},
		{
			label: "Launch mode",
			value: func() string { return flags.LaunchModeLabel(m.settings.LaunchMode) },
			delta: func(dir int) {
				m.settings.LaunchMode = cycle(flags.LaunchModeOptions(), m.settings.LaunchMode, dir)
			},
			note: func() string {
				if m.settings.LaunchMode == "chat" {
					return "chat in terminal after launch"
				}
				return "opens run log & API"
			},
		},
		{
			label: "Show thinking",
			value: func() string { return onOffLabel(!m.settings.HideThinking) },
			delta: func(dir int) { m.settings.HideThinking = !m.settings.HideThinking },
			activate: func() {
				m.settings.HideThinking = !m.settings.HideThinking
			},
		},
		{
			label: "Max tokens",
			value: func() string {
				if m.settings.MaxTokens <= 0 {
					return "unlim"
				}
				return strconv.Itoa(m.settings.MaxTokens)
			},
			slider: func() (float64, bool) { return stopRatio(m.settings.MaxTokens, maxTokenStops), true },
			delta: func(dir int) {
				m.settings.MaxTokens = nextIntStop(m.settings.MaxTokens, dir, maxTokenStops)
			},
			edit: func() string { return strconv.Itoa(m.settings.MaxTokens) },
			apply: func(v string) error {
				t := strings.ToLower(strings.TrimSpace(v))
				if t == "" || t == "0" || t == "unlim" || t == "unlimited" {
					m.settings.MaxTokens = 0
					return nil
				}
				n, err := parseInt(v, 1, 262144)
				if err != nil {
					return fmt.Errorf("max tokens: 0 (unlimited) or 1..262144")
				}
				m.settings.MaxTokens = n
				return nil
			},
		},
		{
			label: "System prompt",
			value: func() string {
				s := strings.TrimSpace(m.settings.SystemPrompt)
				if s == "" {
					return "—"
				}
				return s
			},
			edit: func() string { return m.settings.SystemPrompt },
			apply: func(v string) error {
				m.settings.SystemPrompt = strings.TrimSpace(v)
				return nil
			},
			note: func() string { return "enter to edit" },
		},
		{
			label: "Jinja template",
			value: func() string { return onOffLabel(m.settings.Jinja) },
			delta: func(dir int) { m.settings.Jinja = !m.settings.Jinja },
			activate: func() {
				m.settings.Jinja = !m.settings.Jinja
			},
		},
		{
			label: "Web UI",
			value: func() string {
				if m.settings.NoWebUI {
					return "off (API only)"
				}
				return "on"
			},
			delta: func(dir int) { m.settings.NoWebUI = !m.settings.NoWebUI },
			activate: func() {
				m.settings.NoWebUI = !m.settings.NoWebUI
			},
			note: func() string {
				if m.settings.NoWebUI {
					return "browser page disabled"
				}
				return "browser page enabled"
			},
		},
		{
			label: "Vision projector",
			value: func() string {
				if m.settings.MMProj == "" {
					return "—"
				}
				return filepath.Base(m.settings.MMProj)
			},
			edit: func() string { return m.settings.MMProj },
			apply: func(v string) error {
				v = strings.TrimSpace(strings.Trim(v, `"`))
				if v == "" {
					m.settings.MMProj = ""
					return nil
				}
				if _, err := os.Stat(v); err != nil {
					return fmt.Errorf("projector not found: %s", v)
				}
				m.settings.MMProj = v
				return nil
			},
			note: func() string {
				if m.settings.MMProj == "" {
					return "enter path to mmproj .gguf"
				}
				return "vision enabled"
			},
		},
		{
			label: "Port",
			value: func() string { return fmt.Sprintf("%d", m.settings.Port) },
			delta: func(dir int) { m.settings.Port = clampInt(m.settings.Port+dir, 1024, 65535) },
			edit:  func() string { return strconv.Itoa(m.settings.Port) },
			apply: func(v string) error {
				n, err := parseInt(v, 1, 65535)
				if err != nil {
					return fmt.Errorf("port: 1..65535")
				}
				m.settings.Port = n
				return nil
			},
		},
		{
			label: "Expose on LAN",
			value: func() string {
				if m.settings.LAN {
					return "yes (0.0.0.0)"
				}
				return "no (127.0.0.1)"
			},
			delta:    func(dir int) { m.toggleLAN() },
			activate: func() { m.toggleLAN() },
			note: func() string {
				if m.settings.LAN {
					return "! no API key — exposed on LAN"
				}
				return ""
			},
		},
		{
			label:    "Theme",
			value:    func() string { return m.sty.t.Name },
			slider:   func() (float64, bool) { return listRatio(m.sty.t.Name, ThemeNames()), true },
			delta:    func(dir int) { m.cycleTheme(dir) },
			activate: func() { m.cycleTheme(1) },
			note:     func() string { return fmt.Sprintf("%d palettes", len(AllThemes())) },
		},
		{
			label: "Extra args",
			value: func() string {
				if strings.TrimSpace(m.settings.ExtraArgs) == "" {
					return "—"
				}
				return m.settings.ExtraArgs
			},
			activate: func() {
				m.editing = true
				m.editingField = -1
				m.editingPreset = -1
				m.editingPath = false
				m.editingTitle = "Extra llama-server args"
				m.input.SetValue(m.settings.ExtraArgs)
				m.input.Placeholder = "--cache-reuse 256"
				m.input.Focus()
			},
			note: func() string {
				if m.editing && m.editingField == -1 && !m.editingPath && m.editingPreset < 0 {
					return "editing"
				}
				return "enter to edit"
			},
		},
	}

	// Apply normalisation after any change so the preview is always legal.
	for i := range fs {
		if d := fs[i].delta; d != nil {
			fs[i].delta = func(dir int) {
				d(dir)
				m.settings.Normalize()
				m.clampToModel()
				m.recompute()
			}
		}
		if a := fs[i].activate; a != nil {
			fs[i].activate = func() {
				a()
				m.settings.Normalize()
				m.clampToModel()
				m.recompute()
			}
		}
	}
	return fs
}

// adjustField changes the focused row. The index is into the *visible* rows,
// which is what viewSettings renders and the cursor moves through.
func (m *Model) adjustField(dir int) {
	fs := m.visibleFields()
	if m.fieldIdx < 0 || m.fieldIdx >= len(fs) {
		return
	}
	label := fs[m.fieldIdx].label
	if fs[m.fieldIdx].delta != nil {
		fs[m.fieldIdx].delta(dir)
	}
	m.restoreFieldFocus(label)
}

func (m *Model) restoreFieldFocus(label string) {
	fs := m.visibleFields()
	if len(fs) == 0 {
		m.fieldIdx = 0
		return
	}
	for i, f := range fs {
		if f.label == label {
			m.fieldIdx = i
			return
		}
	}
	if m.fieldIdx >= len(fs) {
		m.fieldIdx = len(fs) - 1
	}
}

// beginEdit opens the text editor for the focused row, if it accepts typed
// input. The index is into the visible rows.
func (m *Model) beginEdit() bool {
	fs := m.visibleFields()
	if m.fieldIdx < 0 || m.fieldIdx >= len(fs) {
		return false
	}
	f := fs[m.fieldIdx]
	if f.apply == nil {
		return false
	}
	m.editing = true
	m.editingPreset = -1
	m.editingPath = false
	m.editingField = m.fieldIdx
	m.editingTitle = f.label
	if f.edit != nil {
		m.input.SetValue(f.edit())
	} else {
		m.input.SetValue("")
	}
	m.input.Placeholder = "type a value, enter to apply"
	m.input.Focus()
	return true
}

// applyTypedValue commits the editor contents to the focused row.
func (m *Model) applyTypedValue(text string) {
	idx := m.editingField
	m.editingField = -1
	m.editingTitle = ""
	fs := m.visibleFields()
	if idx < 0 || idx >= len(fs) || fs[idx].apply == nil {
		return
	}
	label := fs[idx].label
	if err := fs[idx].apply(text); err != nil {
		m.toastf("%s: %v", label, err)
		return
	}
	m.settings.Normalize()
	m.clampToModel()
	m.recompute()
}

func (m *Model) visibleFields() []field {
	var out []field
	for _, f := range m.fields() {
		if f.hidden != nil && f.hidden() {
			continue
		}
		out = append(out, f)
	}
	return out
}

func formatTokensK(tokens int) string {
	if tokens >= 1024 && tokens%1024 == 0 {
		return fmt.Sprintf("%dK", tokens/1024)
	}
	return strconv.Itoa(tokens)
}

func stopRatio(val int, stops []int) float64 {
	if len(stops) < 2 {
		return 0
	}
	if val <= stops[0] {
		return 0
	}
	if val >= stops[len(stops)-1] {
		return 1
	}
	for i := 1; i < len(stops); i++ {
		if val <= stops[i] {
			lo, hi := stops[i-1], stops[i]
			frac := 0.0
			if hi > lo {
				frac = float64(val-lo) / float64(hi-lo)
			}
			return (float64(i-1) + frac) / float64(len(stops)-1)
		}
	}
	return 1
}

func nextIntStop(cur, dir int, stops []int) int {
	if len(stops) == 0 {
		return cur
	}
	if dir > 0 {
		for _, s := range stops {
			if s > cur {
				return s
			}
		}
		return stops[len(stops)-1]
	}
	for i := len(stops) - 1; i >= 0; i-- {
		if stops[i] < cur {
			return stops[i]
		}
	}
	return stops[0]
}

func listRatio(cur string, opts []string) float64 {
	if len(opts) < 2 {
		return 0
	}
	for i, o := range opts {
		if o == cur {
			return float64(i) / float64(len(opts)-1)
		}
	}
	return 0
}

func stepStringList(opts []string, cur string, dir int) string {
	if len(opts) == 0 {
		return cur
	}
	idx := 0
	found := false
	for i, o := range opts {
		if o == cur {
			idx = i
			found = true
			break
		}
	}
	if found {
		idx = clampInt(idx+dir, 0, len(opts)-1)
		return opts[idx]
	}
	if val, err := strconv.Atoi(strings.TrimSpace(cur)); err == nil {
		if dir > 0 {
			for _, o := range opts {
				if n, err := strconv.Atoi(o); err == nil && n > val {
					return o
				}
			}
			return opts[len(opts)-1]
		}
		for i := len(opts) - 1; i >= 0; i-- {
			if n, err := strconv.Atoi(opts[i]); err == nil && n < val {
				return opts[i]
			}
		}
		return opts[0]
	}
	if dir < 0 {
		return opts[0]
	}
	return opts[len(opts)-1]
}

func round2(v float64) float64 {
	return math.Round(v*100) / 100
}

func rangeRatio(val, min, max float64) float64 {
	if max <= min {
		return 0
	}
	return clampFloat((val-min)/(max-min), 0, 1)
}

func onOffLabel(b bool) string {
	if b {
		return "on"
	}
	return "off"
}

func cycle(opts []string, cur string, dir int) string {
	n := len(opts)
	if n == 0 {
		return cur
	}
	idx := 0
	for i, o := range opts {
		if o == cur {
			idx = i
			break
		}
	}
	idx = ((idx+dir)%n + n) % n
	return opts[idx]
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func clampFloat(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func parseInt(v string, min, max int) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return 0, fmt.Errorf("not a whole number: %q", strings.TrimSpace(v))
	}
	if n < min || n > max {
		return 0, fmt.Errorf("%d is outside %d..%d", n, min, max)
	}
	return n, nil
}

func parseFloat(v string, min, max float64) (float64, error) {
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return 0, fmt.Errorf("not a number: %q", strings.TrimSpace(v))
	}
	if f < min || f > max {
		return 0, fmt.Errorf("%g is outside %g..%g", f, min, max)
	}
	return f, nil
}

// toggleLAN flips LAN exposure and asks for confirmation when turning it on.
func (m *Model) toggleLAN() {
	m.settings.LAN = !m.settings.LAN
	if m.settings.LAN {
		m.lanConfirm = true
	}
}

// presetNames lists preset display names.
func presetNames(ps []config.Preset) []string {
	out := make([]string, 0, len(ps))
	for _, p := range ps {
		out = append(out, p.String())
	}
	return out
}

// modelKey is the stable per-model key used for last-used settings.
func modelKey(path string) string { return filepath.Base(path) }

// findProjector looks for a multimodal projector next to the model, so a vision
// model keeps working when the web UI is off.
func findProjector(modelPath string) string {
	dir := filepath.Dir(modelPath)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := strings.ToLower(e.Name())
		if strings.HasPrefix(name, "mmproj") && strings.HasSuffix(name, ".gguf") {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

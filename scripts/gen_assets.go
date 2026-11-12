//go:build ignore

// Command gen_assets rasterizes the alpaca vector mark and wordmark into
// PNG, ICO, social-preview, and Win32 COFF resource (.syso) files using pure Go.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
)

type pt struct{ x, y float64 }

var (
	colBase    = color.NRGBA{0x1e, 0x1e, 0x2e, 0xff}
	colSurface = color.NRGBA{0x31, 0x32, 0x44, 0xff}
	colBorder  = color.NRGBA{0x45, 0x47, 0x5a, 0xff}
	colMauve   = color.NRGBA{0xcb, 0xa6, 0xf7, 0xff}
	colBlue    = color.NRGBA{0x89, 0xb4, 0xfa, 0xff}
	colText    = color.NRGBA{0xcd, 0xd6, 0xf4, 0xff}
	colMuted   = color.NRGBA{0xa6, 0xad, 0xc8, 0xff}
)

var alpacaBody = []pt{
	{58, 224}, {74, 86}, {88, 36}, {108, 70}, {130, 36},
	{142, 78}, {160, 86}, {198, 102}, {200, 130},
	{148, 142}, {132, 172}, {140, 224},
}

var alpacaInnerEar = []pt{{116, 70}, {129, 46}, {136, 76}}
var alpacaNose = []pt{{178, 106}, {198, 108}, {199, 125}, {178, 122}}
var alpacaScarf = []pt{{64, 178}, {134, 162}, {137, 182}, {62, 198}}

func pointInPoly(p pt, poly []pt) bool {
	inside := false
	j := len(poly) - 1
	for i := 0; i < len(poly); i++ {
		if (poly[i].y > p.y) != (poly[j].y > p.y) {
			xint := (poly[j].x-poly[i].x)*(p.y-poly[i].y)/(poly[j].y-poly[i].y) + poly[i].x
			if p.x < xint {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}

func inRoundRect(x, y, rx, ry, rw, rh, rad float64) bool {
	if x < rx || x > rx+rw || y < ry || y > ry+rh {
		return false
	}
	cx := math.Max(rx+rad, math.Min(x, rx+rw-rad))
	cy := math.Max(ry+rad, math.Min(y, ry+rh-rad))
	dx := x - cx
	dy := y - cy
	return dx*dx+dy*dy <= rad*rad
}

func sampleMark(ux, uy float64) color.NRGBA {
	if !inRoundRect(ux, uy, 8, 8, 240, 240, 52) {
		return color.NRGBA{0, 0, 0, 0}
	}
	if !inRoundRect(ux, uy, 14, 14, 228, 228, 46) {
		return colSurface
	}
	p := pt{ux, uy}
	if inRoundRect(ux, uy, 164, 206, 40, 16, 4) {
		return colBlue
	}
	dx, dy := ux-148, uy-104
	if dx*dx+dy*dy <= 9*9 {
		return colBase
	}
	if pointInPoly(p, alpacaInnerEar) || pointInPoly(p, alpacaNose) || pointInPoly(p, alpacaScarf) {
		return colBlue
	}
	if pointInPoly(p, alpacaBody) {
		return colMauve
	}
	return colBase
}

func renderMark(size int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, size, size))
	ss := 6
	if size <= 48 {
		ss = 8
	}
	invSS := 1.0 / float64(ss)
	samples := float64(ss * ss)
	for py := 0; py < size; py++ {
		for px := 0; px < size; px++ {
			var rSum, gSum, bSum, aSum float64
			for sy := 0; sy < ss; sy++ {
				for sx := 0; sx < ss; sx++ {
					ux := (float64(px) + (float64(sx)+0.5)*invSS) * 256.0 / float64(size)
					uy := (float64(py) + (float64(sy)+0.5)*invSS) * 256.0 / float64(size)
					c := sampleMark(ux, uy)
					af := float64(c.A) / 255.0
					rSum += float64(c.R) * af
					gSum += float64(c.G) * af
					bSum += float64(c.B) * af
					aSum += float64(c.A)
				}
			}
			aAvg := aSum / samples
			if aAvg <= 0.5 {
				img.SetNRGBA(px, py, color.NRGBA{})
				continue
			}
			af := aAvg / 255.0
			img.SetNRGBA(px, py, color.NRGBA{
				R: uint8(math.Min(255, rSum/samples/af)),
				G: uint8(math.Min(255, gSum/samples/af)),
				B: uint8(math.Min(255, bSum/samples/af)),
				A: uint8(math.Min(255, aAvg)),
			})
		}
	}
	return img
}

func distSegment(px, py, ax, ay, bx, by float64) float64 {
	dx, dy := bx-ax, by-ay
	l2 := dx*dx + dy*dy
	if l2 == 0 {
		return math.Hypot(px-ax, py-ay)
	}
	t := ((px-ax)*dx + (py-ay)*dy) / l2
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	return math.Hypot(px-(ax+t*dx), py-(ay+t*dy))
}

// distArc computes distance from (px,py) to an axis-aligned circular arc centered at (cx,cy).
func distArc(px, py, cx, cy, r, a0, a1 float64) float64 {
	ang := math.Atan2(py-cy, px-cx)
	norm := func(a float64) float64 {
		for a < 0 {
			a += 2 * math.Pi
		}
		for a >= 2*math.Pi {
			a -= 2 * math.Pi
		}
		return a
	}
	n0, n1, na := norm(a0), norm(a1), norm(ang)
	in := false
	if n0 <= n1 {
		in = na >= n0 && na <= n1
	} else {
		in = na >= n0 || na <= n1
	}
	if in {
		return math.Abs(math.Hypot(px-cx, py-cy) - r)
	}
	d0 := math.Hypot(px-(cx+r*math.Cos(a0)), py-(cy+r*math.Sin(a0)))
	d1 := math.Hypot(px-(cx+r*math.Cos(a1)), py-(cy+r*math.Sin(a1)))
	return math.Min(d0, d1)
}

func sampleWordmark(wx, wy float64) bool {
	const halfW = 6.5
	// a1
	if math.Abs(math.Hypot(wx-194, wy-92)-22) <= halfW ||
		distSegment(wx, wy, 216, 70, 216, 114) <= halfW {
		return true
	}
	// l
	if distSegment(wx, wy, 244, 38, 244, 114) <= halfW {
		return true
	}
	// p
	if distSegment(wx, wy, 274, 70, 274, 136) <= halfW ||
		math.Abs(math.Hypot(wx-296, wy-92)-22) <= halfW {
		return true
	}
	// a2
	if math.Abs(math.Hypot(wx-358, wy-92)-22) <= halfW ||
		distSegment(wx, wy, 380, 70, 380, 114) <= halfW {
		return true
	}
	// c
	if distArc(wx, wy, 424, 92, 22, 0.72, 2*math.Pi-0.72) <= halfW {
		return true
	}
	// a3
	if math.Abs(math.Hypot(wx-486, wy-92)-22) <= halfW ||
		distSegment(wx, wy, 508, 70, 508, 114) <= halfW {
		return true
	}
	return false
}

// Crisp 5x7 pixel bitmap font for the clean tagline on social-preview.png.
var miniFont = map[rune][7]string{
	' ': {"00000", "00000", "00000", "00000", "00000", "00000", "00000"},
	'&': {"01100", "10010", "10100", "01000", "10101", "10010", "01101"},
	'a': {"00000", "00000", "01110", "00001", "01111", "10001", "01111"},
	'b': {"10000", "10000", "11110", "10001", "10001", "10001", "11110"},
	'c': {"00000", "00000", "01110", "10001", "10000", "10001", "01110"},
	'd': {"00001", "00001", "01111", "10001", "10001", "10001", "01111"},
	'e': {"00000", "00000", "01110", "10001", "11111", "10000", "01110"},
	'f': {"00110", "01001", "01000", "11100", "01000", "01000", "01000"},
	'g': {"00000", "00000", "01111", "10001", "01111", "00001", "01110"},
	'h': {"10000", "10000", "11110", "10001", "10001", "10001", "10001"},
	'i': {"00100", "00000", "01100", "00100", "00100", "00100", "01110"},
	'k': {"10000", "10000", "10010", "10100", "11000", "10100", "10010"},
	'l': {"01100", "00100", "00100", "00100", "00100", "00100", "01110"},
	'm': {"00000", "00000", "11010", "10101", "10101", "10001", "10001"},
	'n': {"00000", "00000", "11110", "10001", "10001", "10001", "10001"},
	'o': {"00000", "00000", "01110", "10001", "10001", "10001", "01110"},
	'p': {"00000", "00000", "11110", "10001", "11110", "10000", "10000"},
	'r': {"00000", "00000", "10110", "11001", "10000", "10000", "10000"},
	's': {"00000", "00000", "01111", "10000", "01110", "00001", "11110"},
	't': {"00100", "00100", "11110", "00100", "00100", "00101", "00010"},
	'u': {"00000", "00000", "10001", "10001", "10001", "10011", "01101"},
	'y': {"00000", "00000", "10001", "10001", "01111", "00001", "01110"},
	'.': {"00000", "00000", "00000", "00000", "00000", "01100", "01100"},
	'-': {"00000", "00000", "00000", "11111", "00000", "00000", "00000"},
	'/': {"00001", "00010", "00010", "00100", "01000", "01000", "10000"},
	'A': {"01110", "10001", "10001", "11111", "10001", "10001", "10001"},
	'B': {"11110", "10001", "10001", "11110", "10001", "10001", "11110"},
	'C': {"01110", "10001", "10000", "10000", "10000", "10001", "01110"},
	'D': {"11100", "10010", "10001", "10001", "10001", "10010", "11100"},
	'L': {"10000", "10000", "10000", "10000", "10000", "10000", "11111"},
	'M': {"10001", "11011", "10101", "10101", "10001", "10001", "10001"},
	'P': {"11110", "10001", "10001", "11110", "10000", "10000", "10000"},
	'S': {"01111", "10000", "10000", "01110", "00001", "00001", "11110"},
	'T': {"11111", "00100", "00100", "00100", "00100", "00100", "00100"},
	'U': {"10001", "10001", "10001", "10001", "10001", "10001", "01110"},
	'W': {"10001", "10001", "10001", "10101", "10101", "11011", "10001"},
}

func renderSocialPreview() *image.NRGBA {
	const W, H = 1280, 640
	img := image.NewNRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := colBase
			if !inRoundRect(float64(x), float64(y), 32, 32, W-64, H-64, 36) {
				c = color.NRGBA{0x18, 0x18, 0x25, 0xff}
			} else if !inRoundRect(float64(x), float64(y), 36, 36, W-72, H-72, 32) {
				c = colBorder
			} else if (x%64 == 0 || y%64 == 0) && x > 40 && x < W-40 && y > 40 && y < H-40 {
				c = color.NRGBA{0x24, 0x24, 0x38, 0xff}
			}
			img.SetNRGBA(x, y, c)
		}
	}

	// Render 256px mark cleanly centered on left side (x=150, y=162).
	mark := renderMark(256)
	for py := 0; py < 256; py++ {
		for px := 0; px < 256; px++ {
			s := mark.NRGBAAt(px, py)
			if s.A == 0 {
				continue
			}
			dx, dy := 150+px, 162+py
			bg := img.NRGBAAt(dx, dy)
			af := float64(s.A) / 255.0
			img.SetNRGBA(dx, dy, color.NRGBA{
				R: uint8(float64(s.R)*af + float64(bg.R)*(1-af)),
				G: uint8(float64(s.G)*af + float64(bg.G)*(1-af)),
				B: uint8(float64(s.B)*af + float64(bg.B)*(1-af)),
				A: 0xff,
			})
		}
	}

	// Render crisp antialiased vector wordmark on right (wx in [168..524], scale ~1.72).
	const scale = 1.72
	const ox, oy = 460.0 - 168.0*scale, 255.0 - 90.0*scale
	const ss = 3
	for py := 155; py < 395; py++ {
		for px := 430; px < 1160; px++ {
			hits := 0
			for sy := 0; sy < ss; sy++ {
				for sx := 0; sx < ss; sx++ {
					fx := float64(px) + (float64(sx)+0.5)/float64(ss)
					fy := float64(py) + (float64(sy)+0.5)/float64(ss)
					if sampleWordmark((fx-ox)/scale, (fy-oy)/scale) {
						hits++
					}
				}
			}
			if hits > 0 {
				af := float64(hits) / float64(ss*ss)
				bg := img.NRGBAAt(px, py)
				img.SetNRGBA(px, py, color.NRGBA{
					R: uint8(float64(colText.R)*af + float64(bg.R)*(1-af)),
					G: uint8(float64(colText.G)*af + float64(bg.G)*(1-af)),
					B: uint8(float64(colText.B)*af + float64(bg.B)*(1-af)),
					A: 0xff,
				})
			}
		}
	}

	// Draw accent underline rule.
	for py := 355; py < 361; py++ {
		for px := 460; px < 1180; px++ {
			t := float64(px-460) / float64(1180-460)
			img.SetNRGBA(px, py, color.NRGBA{
				R: uint8(float64(colMauve.R)*(1-t) + float64(colBlue.R)*t),
				G: uint8(float64(colMauve.G)*(1-t) + float64(colBlue.G)*t),
				B: uint8(float64(colMauve.B)*(1-t) + float64(colBlue.B)*t),
				A: 0xff,
			})
		}
	}

	// Render tagline cleanly within [460..1180] with generous right margin.
	drawMiniText(img, 460, 388, 3, "Portable local-LLM launcher for llama.cpp", colText)
	drawMiniText(img, 460, 434, 3, "USB-ready  CUDA & CPU  Tools  MCP  Skills", colBlue)
	return img
}

func drawMiniText(img *image.NRGBA, x0, y0, cell int, text string, col color.NRGBA) {
	cx := x0
	for _, ch := range text {
		glyph, ok := miniFont[ch]
		if !ok {
			cx += 6 * cell
			continue
		}
		for r := 0; r < 7; r++ {
			for c := 0; c < 5; c++ {
				if glyph[r][c] == '1' {
					for dy := 0; dy < cell; dy++ {
						for dx := 0; dx < cell; dx++ {
							img.SetNRGBA(cx+c*cell+dx, y0+r*cell+dy, col)
						}
					}
				}
			}
		}
		cx += 6 * cell
	}
}

func encodePNG(img image.Image) []byte {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		panic(err)
	}
	return buf.Bytes()
}

func encodeBMPFrame(img *image.NRGBA) []byte {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	var buf bytes.Buffer
	// BITMAPINFOHEADER (40 bytes): height is 2*h for XOR + AND masks.
	binary.Write(&buf, binary.LittleEndian, uint32(40))
	binary.Write(&buf, binary.LittleEndian, int32(w))
	binary.Write(&buf, binary.LittleEndian, int32(h*2))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint16(32))
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	binary.Write(&buf, binary.LittleEndian, uint32(w*h*4))
	binary.Write(&buf, binary.LittleEndian, int32(0))
	binary.Write(&buf, binary.LittleEndian, int32(0))
	binary.Write(&buf, binary.LittleEndian, uint32(0))
	binary.Write(&buf, binary.LittleEndian, uint32(0))

	// Bottom-up BGRA pixel rows.
	for y := h - 1; y >= 0; y-- {
		for x := 0; x < w; x++ {
			c := img.NRGBAAt(x, y)
			buf.Write([]byte{c.B, c.G, c.R, c.A})
		}
	}
	// AND mask (1bpp, row padded to 4 bytes, bottom-up).
	andRow := ((w + 31) / 32) * 4
	for y := h - 1; y >= 0; y-- {
		row := make([]byte, andRow)
		for x := 0; x < w; x++ {
			if img.NRGBAAt(x, y).A == 0 {
				row[x/8] |= 1 << (7 - (x % 8))
			}
		}
		buf.Write(row)
	}
	return buf.Bytes()
}

func writeICO(path string, frames []struct {
	size int
	data []byte
}) error {
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint16(0))
	binary.Write(&buf, binary.LittleEndian, uint16(1))
	binary.Write(&buf, binary.LittleEndian, uint16(len(frames)))

	offset := uint32(6 + len(frames)*16)
	for _, f := range frames {
		dim := byte(f.size)
		if f.size >= 256 {
			dim = 0
		}
		buf.WriteByte(dim)
		buf.WriteByte(dim)
		buf.WriteByte(0)
		buf.WriteByte(0)
		binary.Write(&buf, binary.LittleEndian, uint16(1))
		binary.Write(&buf, binary.LittleEndian, uint16(32))
		binary.Write(&buf, binary.LittleEndian, uint32(len(f.data)))
		binary.Write(&buf, binary.LittleEndian, offset)
		offset += uint32(len(f.data))
	}
	for _, f := range frames {
		buf.Write(f.data)
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// buildIconSyso emits a minimal AMD64 COFF object containing .rsrc with RT_ICON (3)
// and RT_GROUP_ICON (14) so CGO_ENABLED=0 `go build` embeds the application icon cleanly.
func buildIconSyso(path string, frames []struct {
	size int
	data []byte
}) error {
	var grp bytes.Buffer
	binary.Write(&grp, binary.LittleEndian, uint16(0))
	binary.Write(&grp, binary.LittleEndian, uint16(1))
	binary.Write(&grp, binary.LittleEndian, uint16(len(frames)))
	for i, f := range frames {
		dim := byte(f.size)
		if f.size >= 256 {
			dim = 0
		}
		grp.WriteByte(dim)
		grp.WriteByte(dim)
		grp.WriteByte(0)
		grp.WriteByte(0)
		binary.Write(&grp, binary.LittleEndian, uint16(1))
		binary.Write(&grp, binary.LittleEndian, uint16(32))
		binary.Write(&grp, binary.LittleEndian, uint32(len(f.data)))
		binary.Write(&grp, binary.LittleEndian, uint16(i+1))
	}

	// Build .rsrc tree: root -> {RT_ICON(3), RT_GROUP_ICON(14)} -> {id 1..N} / {id 1} -> lang 1033 -> data entries -> payloads.
	var rsrc bytes.Buffer
	writeDir := func(nID int) {
		binary.Write(&rsrc, binary.LittleEndian, uint32(0)) // Characteristics
		binary.Write(&rsrc, binary.LittleEndian, uint32(0)) // TimeDateStamp
		binary.Write(&rsrc, binary.LittleEndian, uint16(0)) // MajorVersion
		binary.Write(&rsrc, binary.LittleEndian, uint16(0)) // MinorVersion
		binary.Write(&rsrc, binary.LittleEndian, uint16(0)) // NumberOfNamedEntries
		binary.Write(&rsrc, binary.LittleEndian, uint16(nID))
	}
	// Reserve headers and fill offsets once layout positions are known.
	nIcons := len(frames)
	nRes := nIcons + 1 // N RT_ICON + 1 RT_GROUP_ICON

	rootDirOff := 0
	iconTypeDirOff := 16 + 2*8
	grpTypeDirOff := iconTypeDirOff + 16 + nIcons*8
	iconLangDirsOff := grpTypeDirOff + 16 + 1*8
	grpLangDirOff := iconLangDirsOff + nIcons*(16+8)
	dataEntriesOff := grpLangDirOff + (16 + 8)
	payloadsOff := dataEntriesOff + nRes*16

	// Compute aligned payload offsets.
	payloadOffs := make([]int, nRes)
	payloads := make([][]byte, nRes)
	cur := payloadsOff
	for i, f := range frames {
		payloads[i] = f.data
		payloadOffs[i] = cur
		cur += (len(f.data) + 3) &^ 3
	}
	payloads[nIcons] = grp.Bytes()
	payloadOffs[nIcons] = cur
	cur += (len(payloads[nIcons]) + 3) &^ 3
	totalRsrc := cur

	// 1. Root directory (entries: 3=RT_ICON, 14=RT_GROUP_ICON)
	writeDir(2)
	binary.Write(&rsrc, binary.LittleEndian, uint32(3))
	binary.Write(&rsrc, binary.LittleEndian, uint32(0x80000000|uint32(iconTypeDirOff)))
	binary.Write(&rsrc, binary.LittleEndian, uint32(14))
	binary.Write(&rsrc, binary.LittleEndian, uint32(0x80000000|uint32(grpTypeDirOff)))

	// 2. RT_ICON type directory (entries: 1..N)
	writeDir(nIcons)
	for i := 0; i < nIcons; i++ {
		binary.Write(&rsrc, binary.LittleEndian, uint32(i+1))
		binary.Write(&rsrc, binary.LittleEndian, uint32(0x80000000|uint32(iconLangDirsOff+i*24)))
	}

	// 3. RT_GROUP_ICON type directory (entry: 1)
	writeDir(1)
	binary.Write(&rsrc, binary.LittleEndian, uint32(1))
	binary.Write(&rsrc, binary.LittleEndian, uint32(0x80000000|uint32(grpLangDirOff)))

	// 4. Language directories (1033 = en-US -> data entries)
	for i := 0; i < nRes; i++ {
		writeDir(1)
		binary.Write(&rsrc, binary.LittleEndian, uint32(1033))
		binary.Write(&rsrc, binary.LittleEndian, uint32(dataEntriesOff+i*16))
	}

	// 5. Data entries + relocation offsets (OffsetToData uses ADDR32NB relocation).
	var relocAddrs []uint32
	for i := 0; i < nRes; i++ {
		relocAddrs = append(relocAddrs, uint32(rsrc.Len()))
		binary.Write(&rsrc, binary.LittleEndian, uint32(payloadOffs[i]))
		binary.Write(&rsrc, binary.LittleEndian, uint32(len(payloads[i])))
		binary.Write(&rsrc, binary.LittleEndian, uint32(0)) // CodePage
		binary.Write(&rsrc, binary.LittleEndian, uint32(0)) // Reserved
	}

	// 6. Payloads
	for _, p := range payloads {
		rsrc.Write(p)
		for rsrc.Len()%4 != 0 {
			rsrc.WriteByte(0)
		}
	}
	if rsrc.Len() != totalRsrc {
		return fmt.Errorf("rsrc size mismatch: %d != %d (root=%d)", rsrc.Len(), totalRsrc, rootDirOff)
	}

	const rawDataOffset = 20 + 40
	relocOffset := rawDataOffset + totalRsrc
	symOffset := relocOffset + len(relocAddrs)*10

	var obj bytes.Buffer
	// COFF Header (IMAGE_FILE_MACHINE_AMD64)
	binary.Write(&obj, binary.LittleEndian, uint16(0x8664))
	binary.Write(&obj, binary.LittleEndian, uint16(1))
	binary.Write(&obj, binary.LittleEndian, uint32(0))
	binary.Write(&obj, binary.LittleEndian, uint32(symOffset))
	binary.Write(&obj, binary.LittleEndian, uint32(1)) // 1 symbol (.rsrc)
	binary.Write(&obj, binary.LittleEndian, uint16(0))
	binary.Write(&obj, binary.LittleEndian, uint16(0))

	// Section Header: .rsrc
	var secName [8]byte
	copy(secName[:], ".rsrc")
	obj.Write(secName[:])
	binary.Write(&obj, binary.LittleEndian, uint32(0))
	binary.Write(&obj, binary.LittleEndian, uint32(0))
	binary.Write(&obj, binary.LittleEndian, uint32(totalRsrc))
	binary.Write(&obj, binary.LittleEndian, uint32(rawDataOffset))
	binary.Write(&obj, binary.LittleEndian, uint32(relocOffset))
	binary.Write(&obj, binary.LittleEndian, uint32(0))
	binary.Write(&obj, binary.LittleEndian, uint16(len(relocAddrs)))
	binary.Write(&obj, binary.LittleEndian, uint16(0))
	binary.Write(&obj, binary.LittleEndian, uint32(0x40000040)) // CNT_INITIALIZED_DATA | MEM_READ

	obj.Write(rsrc.Bytes())

	// Relocations (IMAGE_REL_AMD64_ADDR32NB = 3 against symbol 0).
	for _, addr := range relocAddrs {
		binary.Write(&obj, binary.LittleEndian, addr)
		binary.Write(&obj, binary.LittleEndian, uint32(0))
		binary.Write(&obj, binary.LittleEndian, uint16(3))
	}

	// Symbol table: 1 symbol for ".rsrc" section + 4-byte empty string table.
	obj.Write(secName[:])
	binary.Write(&obj, binary.LittleEndian, uint32(0))
	binary.Write(&obj, binary.LittleEndian, int16(1))
	binary.Write(&obj, binary.LittleEndian, uint16(0))
	obj.WriteByte(3) // IMAGE_SYM_CLASS_STATIC
	obj.WriteByte(0)
	binary.Write(&obj, binary.LittleEndian, uint32(4)) // empty string table size

	return os.WriteFile(path, obj.Bytes(), 0o644)
}

func main() {
	if err := os.MkdirAll("assets", 0o755); err != nil {
		panic(err)
	}
	if err := os.MkdirAll("release-evidence", 0o755); err != nil {
		panic(err)
	}

	for _, sz := range []int{512, 256, 64} {
		img := renderMark(sz)
		if err := os.WriteFile(filepath.Join("assets", fmt.Sprintf("logo-%d.png", sz)), encodePNG(img), 0o644); err != nil {
			panic(err)
		}
	}

	// Render ICO frames (BMP for 16/32/48, PNG for 256) and inspect previews in release-evidence.
	var icoFrames []struct {
		size int
		data []byte
	}
	for _, sz := range []int{16, 32, 48, 256} {
		img := renderMark(sz)
		if sz <= 48 {
			_ = os.WriteFile(filepath.Join("release-evidence", fmt.Sprintf("icon-preview-%d.png", sz)), encodePNG(img), 0o644)
			icoFrames = append(icoFrames, struct {
				size int
				data []byte
			}{sz, encodeBMPFrame(img)})
		} else {
			icoFrames = append(icoFrames, struct {
				size int
				data []byte
			}{sz, encodePNG(img)})
		}
	}

	if err := writeICO(filepath.Join("assets", "icon.ico"), icoFrames); err != nil {
		panic(err)
	}
	if err := os.WriteFile(filepath.Join("assets", "social-preview.png"), encodePNG(renderSocialPreview()), 0o644); err != nil {
		panic(err)
	}
	if err := buildIconSyso(filepath.Join("cmd", "alpaca", "alpaca_windows_amd64.syso"), icoFrames); err != nil {
		panic(err)
	}
	fmt.Println("Generated PNGs, icon.ico, social-preview.png, and alpaca_windows_amd64.syso.")
}
